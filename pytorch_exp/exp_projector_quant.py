import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from common.metrics import tensor_metrics
from common.quant_utils import (
    linear_fp16,
    linear_int4_weight_only_fake_quant,
    linear_int8_fake_quant,
    resolve_device,
)
from common.timer import benchmark


def run_variant(name, x, w, b):
    if name == "fp32":
        return F.linear(x, w, b)
    if name == "fp16":
        return linear_fp16(x, w, b)
    if name == "int8_fake_quant":
        return linear_int8_fake_quant(x, w, b)
    if name == "int4_weight_only_fake_quant":
        return linear_int4_weight_only_fake_quant(x, w, b)
    raise ValueError(name)


def check_finite(name, y):
    if not torch.isfinite(y).all():
        raise RuntimeError(f"{name} produced NaN or Inf")


def estimated_weight_size_mb(name, out_dim, in_dim):
    bits = {
        "fp32": 32,
        "fp16": 16,
        "int8_fake_quant": 8,
        "int4_weight_only_fake_quant": 4,
    }[name]
    return out_dim * in_dim * bits / 8 / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    Path("results/csv").mkdir(parents=True, exist_ok=True)

    batch, tokens, in_dim, out_dim = 1, 256, 1152, 2048
    x = torch.randn(batch, tokens, in_dim, device=device)
    w = torch.randn(out_dim, in_dim, device=device) / (in_dim ** 0.5)
    b = torch.randn(out_dim, device=device) * 0.01
    reference = run_variant("fp32", x, w, b)

    rows = []
    for name in ["fp32", "fp16", "int8_fake_quant", "int4_weight_only_fake_quant"]:
        y = run_variant(name, x, w, b)
        check_finite(name, y)
        row = {
            "experiment": "projector_quant",
            "shape": f"batch={batch},tokens={tokens},in={in_dim},out={out_dim}",
            "variant": name,
            "device": device,
            "dtype": str(y.dtype).replace("torch.", ""),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "estimated_weight_size_mb": estimated_weight_size_mb(name, out_dim, in_dim),
        }
        row.update(tensor_metrics(reference, y))
        row.update(benchmark(lambda: run_variant(name, x, w, b), args.repeat, args.warmup, device))
        rows.append(row)

    out_path = Path("results/csv/projector_quant.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
