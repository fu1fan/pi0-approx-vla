import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from common.metrics import tensor_metrics
from common.quant_utils import (
    linear_fp16,
    linear_int4_weight_only_fake_quant,
    linear_int8_fake_quant,
    resolve_device,
)
from common.timer import benchmark, synchronize


CSV_DIR = Path("results/csv")
FIG_DIR = Path("results/figures")
LOG_PATH = Path("research_log.md")


def append_log(message):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(f"\n- [scale_sweep] {message}\n")


def estimated_weight_size_mb(name, out_dim, in_dim):
    bits = {
        "fp32": 32,
        "fp16": 16,
        "int8_fake_quant": 8,
        "int4_weight_only_fake_quant": 4,
    }[name]
    return out_dim * in_dim * bits / 8 / (1024 * 1024)


def run_linear_variant(name, x, w, b):
    if name == "fp32":
        return F.linear(x, w, b)
    if name == "fp16":
        return linear_fp16(x, w, b)
    if name == "int8_fake_quant":
        return linear_int8_fake_quant(x, w, b)
    if name == "int4_weight_only_fake_quant":
        return linear_int4_weight_only_fake_quant(x, w, b)
    raise ValueError(name)


def stable_x(x):
    return x - x.max(dim=-1, keepdim=True).values


def lut_exp(x, x_min=-8.0, x_max=0.0, entries=256):
    table_x = torch.linspace(x_min, x_max, entries, device=x.device)
    table_y = torch.exp(table_x)
    pos = (x.clamp(x_min, x_max) - x_min) / (x_max - x_min) * (entries - 1)
    idx0 = torch.floor(pos).long().clamp(0, entries - 1)
    idx1 = (idx0 + 1).clamp(0, entries - 1)
    frac = pos - idx0.float()
    return table_y[idx0] * (1.0 - frac) + table_y[idx1] * frac


def pwl_exp(x):
    y = torch.zeros_like(x)
    xc = x.clamp(-8.0, 0.0)
    for lo, hi in [(-8.0, -6.0), (-6.0, -4.0), (-4.0, -2.0), (-2.0, -1.0), (-1.0, -0.5), (-0.5, 0.0)]:
        mask = (xc >= lo) & (xc <= hi)
        y0 = torch.exp(torch.tensor(lo, device=x.device))
        y1 = torch.exp(torch.tensor(hi, device=x.device))
        slope = (y1 - y0) / (hi - lo)
        y[mask] = y0 + slope * (xc[mask] - lo)
    return y


def taylor_exp(x, order):
    xc = x.clamp(-8.0, 0.0)
    scale = torch.tensor(8.0, device=x.device, dtype=x.dtype)
    z = xc / scale
    y = 1.0 + z + 0.5 * z * z
    if order >= 3:
        y = y + (z * z * z) / 6.0
    return y.clamp_min(1e-6).pow(scale)


def normalize(exp_x):
    return exp_x / exp_x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def run_softmax_variant(name, x):
    sx = stable_x(x)
    if name == "exact_softmax":
        return torch.softmax(sx, dim=-1)
    if name == "lut_exp_softmax":
        return normalize(lut_exp(sx))
    if name == "pwl_exp_softmax":
        return normalize(pwl_exp(sx))
    if name == "taylor2_exp_softmax":
        return normalize(taylor_exp(sx, 2))
    if name == "taylor3_exp_softmax":
        return normalize(taylor_exp(sx, 3))
    raise ValueError(name)


def check_finite(name, y):
    if not torch.isfinite(y).all():
        raise RuntimeError(f"{name} produced NaN or Inf")


def clear_device_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()


def is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def repeat_for_linear(batch, seq, dim, requested):
    if dim >= 8192:
        return min(requested, 10)
    if dim >= 4096:
        return min(requested, 20)
    return requested


def repeat_for_softmax(heads, seq, requested):
    elements = heads * seq * seq
    if elements >= 32 * 1024 * 1024:
        return min(requested, 5)
    if elements >= 8 * 1024 * 1024:
        return min(requested, 10)
    return requested


def run_linear_sweep(args, device):
    variants = ["fp32", "fp16", "int8_fake_quant", "int4_weight_only_fake_quant"]
    shapes = [(1, 256, 2048, 2048), (1, 256, 4096, 4096), (1, 128, 8192, 8192)]
    rows = []
    for batch, seq, in_dim, out_dim in shapes:
        shape_label = f"batch={batch},seq={seq},in={in_dim},out={out_dim}"
        repeat = repeat_for_linear(batch, seq, in_dim, args.repeat)
        if repeat < args.repeat:
            append_log(f"linear {shape_label}: reduced repeat from {args.repeat} to {repeat} to keep runtime bounded.")
        try:
            x = torch.randn(batch, seq, in_dim, device=device)
            w = torch.randn(out_dim, in_dim, device=device) / (in_dim ** 0.5)
            b = torch.randn(out_dim, device=device) * 0.01
            reference = run_linear_variant("fp32", x, w, b)
            fp32_size = estimated_weight_size_mb("fp32", out_dim, in_dim)
            shape_rows = []
            for name in variants:
                try:
                    y = run_linear_variant(name, x, w, b)
                    check_finite(name, y)
                    size_mb = estimated_weight_size_mb(name, out_dim, in_dim)
                    row = {
                        "experiment": "scale_sweep_linear",
                        "shape": shape_label,
                        "batch": batch,
                        "seq": seq,
                        "in_dim": in_dim,
                        "out_dim": out_dim,
                        "variant": name,
                        "device": device,
                        "dtype": str(y.dtype).replace("torch.", ""),
                        "repeat": repeat,
                        "warmup": args.warmup,
                        "estimated_weight_size_MB": size_mb,
                        "compression_ratio_vs_fp32": fp32_size / size_mb,
                    }
                    row.update(tensor_metrics(reference, y))
                    row.update(benchmark(lambda n=name: run_linear_variant(n, x, w, b), repeat, args.warmup, device))
                    shape_rows.append(row)
                    del y
                    clear_device_cache(device)
                except RuntimeError as exc:
                    clear_device_cache(device)
                    if is_oom(exc):
                        append_log(f"linear {shape_label} variant {name}: OOM, skipped variant.")
                        continue
                    raise
            if shape_rows:
                fp32_latency = next((r["latency_mean_ms"] for r in shape_rows if r["variant"] == "fp32"), None)
                for row in shape_rows:
                    row["speedup_vs_fp32"] = (fp32_latency / row["latency_mean_ms"]) if fp32_latency else float("nan")
                rows.extend(shape_rows)
        except RuntimeError as exc:
            clear_device_cache(device)
            if is_oom(exc):
                append_log(f"linear {shape_label}: OOM, skipped shape.")
                continue
            raise
    return rows


def run_softmax_sweep(args, device):
    variants = ["exact_softmax", "lut_exp_softmax", "pwl_exp_softmax", "taylor2_exp_softmax", "taylor3_exp_softmax"]
    shapes = [(8, 128), (8, 256), (16, 512), (32, 1024)]
    rows = []
    for heads, seq in shapes:
        shape_label = f"heads={heads},seq={seq}"
        repeat = repeat_for_softmax(heads, seq, args.repeat)
        if repeat < args.repeat:
            append_log(f"softmax {shape_label}: reduced repeat from {args.repeat} to {repeat} to keep runtime bounded.")
        start = time.perf_counter()
        try:
            x = torch.randn(heads, seq, seq, device=device)
            reference = run_softmax_variant("exact_softmax", x)
            shape_rows = []
            for name in variants:
                try:
                    y = run_softmax_variant(name, x)
                    check_finite(name, y)
                    row = {
                        "experiment": "scale_sweep_softmax",
                        "shape": shape_label,
                        "heads": heads,
                        "seq": seq,
                        "variant": name,
                        "device": device,
                        "dtype": str(y.dtype).replace("torch.", ""),
                        "repeat": repeat,
                        "warmup": args.warmup,
                    }
                    row.update(tensor_metrics(reference, y, include_kl=True))
                    row.update(benchmark(lambda n=name: run_softmax_variant(n, x), repeat, args.warmup, device))
                    shape_rows.append(row)
                    del y
                    clear_device_cache(device)
                except RuntimeError as exc:
                    clear_device_cache(device)
                    if is_oom(exc):
                        append_log(f"softmax {shape_label} variant {name}: OOM, skipped variant.")
                        continue
                    raise
            exact_latency = next((r["latency_mean_ms"] for r in shape_rows if r["variant"] == "exact_softmax"), None)
            for row in shape_rows:
                row["speedup_vs_exact"] = (exact_latency / row["latency_mean_ms"]) if exact_latency else float("nan")
            rows.extend(shape_rows)
            elapsed = time.perf_counter() - start
            if elapsed > args.slow_shape_seconds:
                append_log(f"softmax {shape_label}: completed but took {elapsed:.1f}s; larger repeats should be avoided.")
        except RuntimeError as exc:
            clear_device_cache(device)
            if is_oom(exc):
                append_log(f"softmax {shape_label}: OOM, skipped shape.")
                continue
            raise
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        append_log(f"{path}: no rows generated.")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_scale_results(linear_path, softmax_path):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    linear = pd.DataFrame()
    softmax = pd.DataFrame()
    if linear_path.exists():
        linear = pd.read_csv(linear_path)
        frames.append(linear.assign(panel="linear"))
    if softmax_path.exists():
        softmax = pd.read_csv(softmax_path)
        frames.append(softmax.assign(panel="softmax"))

    if frames:
        latency_df = pd.concat(frames, ignore_index=True)
        labels = latency_df["panel"] + "\n" + latency_df["variant"] + "\n" + latency_df["shape"].astype(str)
        plt.figure(figsize=(max(10, 0.45 * len(latency_df)), 5.5))
        plt.bar(range(len(latency_df)), latency_df["latency_mean_ms"])
        plt.xticks(range(len(latency_df)), labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("latency_mean_ms")
        plt.title("Scale Sweep Latency")
        plt.tight_layout()
        out = FIG_DIR / "scale_sweep_latency.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"wrote {out}")

    if not linear.empty:
        memory_df = linear.copy()
        labels = memory_df["variant"] + "\n" + memory_df["shape"].astype(str)
        plt.figure(figsize=(max(10, 0.45 * len(memory_df)), 5.5))
        plt.bar(range(len(memory_df)), memory_df["estimated_weight_size_MB"])
        plt.xticks(range(len(memory_df)), labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("estimated_weight_size_MB")
        plt.title("Scale Sweep Linear Weight Memory")
        plt.tight_layout()
        out = FIG_DIR / "scale_sweep_memory.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"wrote {out}")

    error_frames = []
    if not linear.empty:
        error_frames.append(linear[linear["variant"].ne("fp32")].assign(panel="linear"))
    if not softmax.empty:
        error_frames.append(softmax[softmax["variant"].ne("exact_softmax")].assign(panel="softmax"))
    if error_frames:
        error_df = pd.concat(error_frames, ignore_index=True)
        labels = error_df["panel"] + "\n" + error_df["variant"] + "\n" + error_df["shape"].astype(str)
        plt.figure(figsize=(max(10, 0.45 * len(error_df)), 5.5))
        plt.bar(range(len(error_df)), error_df["mse"])
        plt.yscale("log")
        plt.xticks(range(len(error_df)), labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("MSE (log scale)")
        plt.title("Scale Sweep Error")
        plt.tight_layout()
        out = FIG_DIR / "scale_sweep_error.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slow-shape-seconds", type=float, default=60.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    append_log(f"started scale sweep on device={device}, requested repeat={args.repeat}, warmup={args.warmup}.")
    if device == "cuda":
        append_log(f"cuda device: {torch.cuda.get_device_name(0)}.")

    linear_rows = run_linear_sweep(args, device)
    softmax_rows = run_softmax_sweep(args, device)

    linear_path = CSV_DIR / "scale_sweep_linear.csv"
    softmax_path = CSV_DIR / "scale_sweep_softmax.csv"
    write_csv(linear_path, linear_rows)
    write_csv(softmax_path, softmax_rows)
    plot_scale_results(linear_path, softmax_path)
    append_log(f"completed scale sweep: {len(linear_rows)} linear rows, {len(softmax_rows)} softmax rows.")
    synchronize(device)


if __name__ == "__main__":
    main()
