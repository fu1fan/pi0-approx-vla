import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common.metrics import tensor_metrics
from common.quant_utils import (
    fake_quant_symmetric,
    fake_quant_weight_per_out_channel,
    resolve_device,
)
from common.timer import benchmark


VARIANTS = ["fp32", "fp16", "int8_fake_quant", "int4_weight_only_fake_quant", "w4a8_fake_quant"]


def append_research_log(message, path=Path("research_log.md")):
    with path.open("a") as f:
        f.write(f"\n- {message}\n")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cache(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} produced NaN or Inf")


def tensor_metrics_with_relative_l2(reference, candidate, include_kl=False):
    metrics = tensor_metrics(reference, candidate, include_kl=include_kl)
    ref = reference.detach().float().reshape(-1)
    out = candidate.detach().float().reshape(-1)
    metrics["relative_l2_error"] = (torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-12)).item()
    return metrics


def dtype_name(tensor):
    return str(tensor.dtype).replace("torch.", "")


def linear_variant(x, w, b, variant):
    if variant == "fp32":
        return F.linear(x.float(), w.float(), b.float() if b is not None else None)
    if variant == "fp16":
        return F.linear(x.half(), w.half(), b.half() if b is not None else None).float()
    if variant == "int8_fake_quant":
        xq, _ = fake_quant_symmetric(x, 8)
        wq, _ = fake_quant_symmetric(w, 8)
        return F.linear(xq, wq, b)
    if variant == "int4_weight_only_fake_quant":
        wq, _ = fake_quant_weight_per_out_channel(w, 4)
        return F.linear(x, wq, b)
    if variant == "w4a8_fake_quant":
        xq, _ = fake_quant_symmetric(x, 8)
        wq, _ = fake_quant_weight_per_out_channel(w, 4)
        return F.linear(xq, wq, b)
    raise ValueError(variant)


def gelu_variant(x, activation):
    if activation == "gelu":
        return F.gelu(x)
    if activation == "tanh_gelu":
        return F.gelu(x, approximate="tanh")
    if activation == "identity":
        return x
    raise ValueError(activation)


def gated_ffn_variant(x, gate_w, up_w, down_w, gate_b, up_b, down_b, variant, activation="gelu"):
    gate = linear_variant(x, gate_w, gate_b, variant)
    up = linear_variant(x, up_w, up_b, variant)
    hidden = gelu_variant(gate, activation) * up
    return linear_variant(hidden, down_w, down_b, variant)


def weight_size_mb(num_params, variant):
    bits = {
        "fp32": 32,
        "fp16": 16,
        "int8_fake_quant": 8,
        "int4_weight_only_fake_quant": 4,
        "w4a8_fake_quant": 4,
    }[variant]
    return num_params * bits / 8 / (1024 * 1024)


def make_linear_weight(out_dim, in_dim, device):
    w = torch.randn(out_dim, in_dim, device=device) / (in_dim ** 0.5)
    b = torch.randn(out_dim, device=device) * 0.01
    return w, b


def make_ffn_weights(in_dim, mlp_dim, out_dim, device):
    gate_w, gate_b = make_linear_weight(mlp_dim, in_dim, device)
    up_w, up_b = make_linear_weight(mlp_dim, in_dim, device)
    down_w, down_b = make_linear_weight(out_dim, mlp_dim, device)
    return gate_w, up_w, down_w, gate_b, up_b, down_b


def safe_benchmark(fn, repeat, warmup, device):
    return benchmark(fn, repeat=repeat, warmup=warmup, device=device)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def bar_plot(rows, metric, out_path, title, log_y=False):
    if not rows:
        return
    values = [row.get(metric) for row in rows]
    plot_rows = [(row, value) for row, value in zip(rows, values) if value is not None]
    if not plot_rows:
        return
    labels = [f"{row['module']}\n{row['variant']}\n{row.get('shape', '')}" for row, _ in plot_rows]
    vals = [value for _, value in plot_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, min(28, 0.35 * len(vals))), 6))
    plt.bar(range(len(vals)), vals)
    if log_y:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"wrote {out_path}")


def timed_stage(label):
    class Timer:
        def __enter__(self_inner):
            self_inner.start = time.perf_counter()
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            elapsed = time.perf_counter() - self_inner.start
            print(f"{label}: {elapsed:.2f}s")

    return Timer()
