#!/usr/bin/env python3
"""Parse HLS run statuses and copied Vitis reports into a summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDS = [
    "kernel",
    "variant",
    "comparison_group",
    "role",
    "data_type",
    "shape",
    "csim_status",
    "hls_csim_status",
    "hls_synth_status",
    "parse_status",
    "latency_cycles",
    "ii",
    "estimated_clock_ns",
    "LUT",
    "FF",
    "BRAM",
    "DSP",
    "mse",
    "mae",
    "kl",
    "cosine",
    "relative_l2",
    "notes",
    "report_path",
]


KERNEL_METADATA = {
    "int8_gemm": {"comparison_group": "gemm", "role": "optimized"},
    "exact_softmax": {"comparison_group": "softmax", "role": "baseline"},
    "lut_softmax": {"comparison_group": "softmax", "role": "optimized"},
    "exact_gelu": {"comparison_group": "gelu", "role": "baseline"},
    "gelu_pwl": {"comparison_group": "gelu", "role": "optimized"},
    "exact_rmsnorm": {"comparison_group": "rmsnorm", "role": "baseline"},
    "rmsnorm_rsqrt": {"comparison_group": "rmsnorm", "role": "optimized"},
    "fixed_projector_tile": {"comparison_group": "projector", "role": "optimized"},
}


def text_or_blank(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def load_status(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_xml_value(root: ET.Element, names: tuple[str, ...]) -> str:
    wanted = {name.lower().replace("_", "").replace("-", "") for name in names}
    for elem in root.iter():
        normalized = elem.tag.lower().split("}")[-1].replace("_", "").replace("-", "")
        if normalized in wanted and elem.text:
            return elem.text.strip()
    return ""


def parse_xml_report(path: Path) -> dict[str, str]:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {}
    return {
        "latency_cycles": find_xml_value(root, ("Worst-caseLatency", "Average-caseLatency", "Latency")),
        "ii": find_xml_value(root, ("Interval-max", "Interval", "PipelineII")),
        "estimated_clock_ns": find_xml_value(root, ("EstimatedClockPeriod", "EstimatedClock")),
        "LUT": find_xml_value(root, ("LUT",)),
        "FF": find_xml_value(root, ("FF",)),
        "BRAM": find_xml_value(root, ("BRAM_18K", "BRAM")),
        "DSP": find_xml_value(root, ("DSP", "DSP48E")),
    }


def parse_text_report(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    latency_match = re.search(r"\|\s*Latency\s*\(cycles\)\s*\|[^\n]*\n\|[-\s|:]+\|\n\|\s*([\d]+)\s*\|\s*([\d]+)", text)
    if latency_match:
        out["latency_cycles"] = latency_match.group(2)
    interval_match = re.search(r"\|\s*Interval\s*\|[^\n]*\n\|[-\s|:]+\|\n\|\s*([\d]+)\s*\|\s*([\d]+)", text)
    if interval_match:
        out["ii"] = interval_match.group(2)
    clock_match = re.search(r"Estimated\s+(?:Clock\s+)?Period\s*[:=]?\s*([\d.]+)", text, flags=re.IGNORECASE)
    if clock_match:
        out["estimated_clock_ns"] = clock_match.group(1)
    for name, field in (("LUT", "LUT"), ("FF", "FF"), ("BRAM", "BRAM"), ("DSP", "DSP")):
        match = re.search(rf"\|\s*{name}(?:_18K|48E)?\s*\|\s*([\d.]+)", text)
        if match:
            out[field] = match.group(1)
    return out


def parse_reports(kernel_dir: Path) -> tuple[dict[str, str], str, str]:
    candidates = sorted((kernel_dir / "raw_reports").rglob("*csynth*.xml"))
    for path in candidates:
        parsed = {k: v for k, v in parse_xml_report(path).items() if v}
        if parsed:
            return parsed, "ok", path.as_posix()
    candidates = sorted((kernel_dir / "raw_reports").rglob("*.rpt"))
    for path in candidates:
        parsed = {k: v for k, v in parse_text_report(path).items() if v}
        if parsed:
            return parsed, "ok", path.as_posix()
    return {}, "parse_failed", ""


def metric_rows(status: dict) -> list[dict[str, str]]:
    hls_metrics = status.get("hls", {}).get("csim", {}).get("metrics", [])
    local_metrics = status.get("local_csim", {}).get("metrics", [])
    metrics = hls_metrics or local_metrics or [{}]
    return metrics


def make_rows(reports_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for status_path in sorted(reports_dir.glob("*/run_status.json")):
        status = load_status(status_path)
        kernel_dir = status_path.parent
        kernel_name = status.get("kernel", "")
        metadata = KERNEL_METADATA.get(kernel_name, {})
        report_values, parse_status, report_path = parse_reports(kernel_dir)
        synth_status = status.get("hls", {}).get("synthesis", {}).get("status", "")
        if synth_status not in {"passed"} and not report_values:
            parse_status = "no_synthesis_report"
        for metric in metric_rows(status):
            row = {field: "" for field in FIELDS}
            row.update(
                {
                    "kernel": status.get("kernel", ""),
                    "variant": text_or_blank(metric.get("variant", "default")),
                    "comparison_group": text_or_blank(
                        metric.get("comparison_group", status.get("comparison_group", metadata.get("comparison_group", "")))
                    ),
                    "role": text_or_blank(metric.get("role", status.get("role", metadata.get("role", "")))),
                    "data_type": text_or_blank(metric.get("dtype", status.get("dtype", ""))),
                    "shape": text_or_blank(metric.get("shape", status.get("shape", ""))),
                    "csim_status": status.get("local_csim", {}).get("status", ""),
                    "hls_csim_status": status.get("hls", {}).get("csim", {}).get("status", ""),
                    "hls_synth_status": synth_status,
                    "parse_status": parse_status,
                    "mse": text_or_blank(metric.get("mse", "")),
                    "mae": text_or_blank(metric.get("mae", "")),
                    "kl": text_or_blank(metric.get("kl", "")),
                    "cosine": text_or_blank(metric.get("cosine", "")),
                    "relative_l2": text_or_blank(metric.get("relative_l2", "")),
                    "report_path": report_path,
                }
            )
            for key, value in report_values.items():
                row[key] = value
            if synth_status not in {"passed", ""}:
                row["notes"] = f"HLS synthesis {synth_status}; see results/hls_reports/{status.get('kernel')}/vitis_synthesis_output.txt"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="results/hls_reports", type=Path)
    parser.add_argument("--out", default="results/csv/hls_kernel_summary.csv", type=Path)
    args = parser.parse_args()

    rows = make_rows(args.reports_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
