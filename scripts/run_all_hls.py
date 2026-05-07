#!/usr/bin/env python3
"""Run local C-sim fallbacks and Vitis HLS synthesis attempts for all kernels."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


KERNELS = [
    {
        "kernel": "int8_gemm",
        "component": "vitis_workspace/int8_gemm",
        "src": "vitis_workspace/hls_src/int8_gemm",
        "shape": "50x32x1024",
        "dtype": "int8_acc32_out16",
        "comparison_group": "gemm",
        "role": "optimized",
        "optional": False,
    },
    {
        "kernel": "exact_softmax",
        "component": "vitis_workspace/exact_softmax",
        "src": "vitis_workspace/hls_src/exact_softmax",
        "shape": "rows4_len128",
        "dtype": "float32",
        "comparison_group": "softmax",
        "role": "baseline",
        "optional": False,
    },
    {
        "kernel": "lut_softmax",
        "component": "vitis_workspace/lut_softmax",
        "src": "vitis_workspace/hls_src/lut_softmax",
        "shape": "rows4_len128",
        "dtype": "fixed16x6_prob18x2",
        "comparison_group": "softmax",
        "role": "optimized",
        "optional": False,
    },
    {
        "kernel": "exact_gelu",
        "component": "vitis_workspace/exact_gelu",
        "src": "vitis_workspace/hls_src/exact_gelu",
        "shape": "len4096",
        "dtype": "float32",
        "comparison_group": "gelu",
        "role": "baseline",
        "optional": False,
    },
    {
        "kernel": "gelu_pwl",
        "component": "vitis_workspace/gelu_pwl",
        "src": "vitis_workspace/hls_src/gelu_pwl",
        "shape": "len4096",
        "dtype": "fixed16x6",
        "comparison_group": "gelu",
        "role": "optimized",
        "optional": False,
    },
    {
        "kernel": "exact_rmsnorm",
        "component": "vitis_workspace/exact_rmsnorm",
        "src": "vitis_workspace/hls_src/exact_rmsnorm",
        "shape": "hidden1024",
        "dtype": "float32",
        "comparison_group": "rmsnorm",
        "role": "baseline",
        "optional": False,
    },
    {
        "kernel": "rmsnorm_rsqrt",
        "component": "vitis_workspace/rmsnorm_rsqrt",
        "src": "vitis_workspace/hls_src/rmsnorm_rsqrt",
        "shape": "hidden1024",
        "dtype": "fixed16x6_acc40x16",
        "comparison_group": "rmsnorm",
        "role": "optimized",
        "optional": False,
    },
    {
        "kernel": "fixed_projector_tile",
        "component": "vitis_workspace/fixed_projector_tile",
        "src": "vitis_workspace/hls_src/fixed_projector_tile",
        "shape": "64x1152x256",
        "dtype": "fixed16x6_acc40x16",
        "comparison_group": "projector",
        "role": "optimized",
        "optional": True,
    },
]


METRIC_RE = re.compile(r"^HLS_METRIC\s+(?P<body>.*)$")


def parse_metric_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        match = METRIC_RE.match(line.strip())
        if not match:
            continue
        row: dict[str, str] = {}
        for token in match.group("body").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            row[key] = value
        rows.append(row)
    return rows


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_sec: int,
    env: dict[str, str] | None = None,
) -> dict:
    started = datetime.now().isoformat(timespec="seconds")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=timeout_sec)
        return {
            "command": cmd,
            "cwd": str(cwd),
            "started_at": started,
            "returncode": proc.returncode,
            "status": "passed" if proc.returncode == 0 else "failed",
            "stdout": stdout,
            "timeout_sec": timeout_sec,
        }
    except subprocess.TimeoutExpired:
        output = ""
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                output, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = proc.communicate()
        return {
            "command": cmd,
            "cwd": str(cwd),
            "started_at": started,
            "returncode": None,
            "status": "timeout",
            "stdout": output,
            "timeout_sec": timeout_sec,
        }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_local_csim(kernel: dict, report_dir: Path, timeout_sec: int) -> dict:
    src = Path(kernel["src"]).resolve()
    binary = Path("/tmp") / f"pi0_hls_{kernel['kernel']}_tb"
    compile_cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-DHLS_NO_AP_FIXED",
        "kernel.cpp",
        "tb.cpp",
        "-o",
        str(binary),
    ]
    compile_result = run_command(compile_cmd, src, timeout_sec)
    run_result = {
        "status": "not_run",
        "stdout": "",
        "returncode": None,
        "command": [str(binary)],
        "cwd": str(src),
    }
    if compile_result["status"] == "passed":
        run_result = run_command([str(binary)], src, timeout_sec)

    combined = (
        "$ " + " ".join(compile_cmd) + "\n" + compile_result.get("stdout", "") + "\n" +
        "$ " + str(binary) + "\n" + run_result.get("stdout", "")
    )
    write_text(report_dir / "local_csim_output.txt", combined)
    metrics = parse_metric_lines(run_result.get("stdout", ""))
    return {
        "status": run_result["status"] if compile_result["status"] == "passed" else "compile_failed",
        "compile_status": compile_result["status"],
        "run_status": run_result["status"],
        "metrics": metrics,
        "output": "local_csim_output.txt",
    }


def run_python_golden(kernel: dict, report_dir: Path, timeout_sec: int) -> dict:
    src = Path(kernel["src"]).resolve()
    golden_scripts = sorted(src.glob("golden_*.py"))
    if not golden_scripts:
        return {"status": "missing", "metrics": []}
    result = run_command(["python", golden_scripts[0].name], src, timeout_sec)
    write_text(report_dir / "python_golden_output.txt", result.get("stdout", ""))
    metrics: list[dict] = []
    try:
        parsed = json.loads(result.get("stdout", ""))
        metrics = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        metrics = []
    return {
        "status": result["status"],
        "script": golden_scripts[0].name,
        "metrics": metrics,
        "output": "python_golden_output.txt",
    }


def hls_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XILINX_VITIS_DATA_DIR", "/tmp/vitis_data")
    Path(env["XILINX_VITIS_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def hls_commands() -> tuple[list[str] | None, list[str] | None, str]:
    csim_cmd = None
    tool = "missing"
    if shutil.which("vitis-run"):
        csim_cmd = ["vitis-run", "--mode", "hls", "--config", "hls_config.cfg", "--work_dir", "build", "--csim"]
        tool = "vitis-run"
    if shutil.which("v++"):
        synth_cmd = ["v++", "--compile", "--mode", "hls", "--config", "hls_config.cfg"]
        return csim_cmd, synth_cmd, f"{tool}+v++" if csim_cmd else "v++"
    return csim_cmd, None, tool


def copy_hls_reports(component: Path, report_dir: Path) -> list[str]:
    copied: list[str] = []
    raw_dir = report_dir / "raw_reports"
    for path in component.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".rpt", ".xml"}:
            continue
        if ".cache" in path.parts or ".autopilot" in path.parts or "logs" in path.parts:
            continue
        if "syn" in path.parts and "report" not in path.parts:
            continue
        rel = path.relative_to(component)
        dst = raw_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(dst.relative_to(report_dir).as_posix())
    return copied


def run_hls(kernel: dict, report_dir: Path, csim_timeout_sec: int, synth_timeout_sec: int) -> dict:
    component = Path(kernel["component"]).resolve()
    csim_cmd, synth_cmd, tool = hls_commands()
    status = {
        "tool": tool,
        "csim": {"status": "not_run"},
        "synthesis": {"status": "not_run"},
        "copied_reports": [],
    }
    if synth_cmd is None:
        status["synthesis"] = {"status": "tool_missing", "reason": "Neither vitis-run nor v++ found"}
        return status

    env = hls_env()
    if csim_cmd is not None:
        csim = run_command(csim_cmd, component, csim_timeout_sec, env=env)
        write_text(report_dir / "vitis_csim_output.txt", csim.get("stdout", ""))
        status["csim"] = {
            "status": csim["status"],
            "returncode": csim["returncode"],
            "command": csim["command"],
            "output": "vitis_csim_output.txt",
            "metrics": parse_metric_lines(csim.get("stdout", "")),
        }

    synth = run_command(synth_cmd, component, synth_timeout_sec, env=env)
    write_text(report_dir / "vitis_synthesis_output.txt", synth.get("stdout", ""))
    status["synthesis"] = {
        "status": synth["status"],
        "returncode": synth["returncode"],
        "command": synth["command"],
        "output": "vitis_synthesis_output.txt",
    }
    status["copied_reports"] = copy_hls_reports(component, report_dir)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="results/hls_reports", type=Path)
    parser.add_argument("--kernels", nargs="*", default=None)
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--no-synthesis", action="store_true")
    parser.add_argument("--local-timeout-sec", type=int, default=120)
    parser.add_argument("--csim-timeout-sec", type=int, default=300)
    parser.add_argument("--synth-timeout-sec", type=int, default=600)
    args = parser.parse_args()

    selected = []
    requested = set(args.kernels or [])
    for kernel in KERNELS:
        if requested and kernel["kernel"] not in requested:
            continue
        if args.skip_optional and kernel["optional"]:
            continue
        selected.append(kernel)

    for kernel in selected:
        report_dir = args.reports_dir / kernel["kernel"]
        report_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "kernel": kernel["kernel"],
            "shape": kernel["shape"],
            "dtype": kernel["dtype"],
            "comparison_group": kernel.get("comparison_group", ""),
            "role": kernel.get("role", ""),
            "optional": kernel["optional"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "local_csim": run_local_csim(kernel, report_dir, args.local_timeout_sec),
            "python_golden": run_python_golden(kernel, report_dir, args.local_timeout_sec),
        }
        if args.no_synthesis:
            status["hls"] = {"synthesis": {"status": "skipped_by_user"}}
        else:
            status["hls"] = run_hls(kernel, report_dir, args.csim_timeout_sec, args.synth_timeout_sec)
        (report_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(
            f"{kernel['kernel']}: local={status['local_csim']['status']} hls={status['hls'].get('synthesis', {}).get('status')}",
            flush=True,
        )

    all_status: dict[str, dict] = {}
    for status_path in sorted(args.reports_dir.glob("*/run_status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        all_status[status.get("kernel", status_path.parent.name)] = status

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "hls_run_manifest.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "kernels": all_status}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
