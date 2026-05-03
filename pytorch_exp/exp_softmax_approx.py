import argparse
import csv
from pathlib import Path

import torch

from common.metrics import tensor_metrics
from common.quant_utils import resolve_device
from common.timer import benchmark


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
    segments = [
        (-8.0, -6.0),
        (-6.0, -4.0),
        (-4.0, -2.0),
        (-2.0, -1.0),
        (-1.0, -0.5),
        (-0.5, 0.0),
    ]
    xc = x.clamp(-8.0, 0.0)
    for lo, hi in segments:
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
    y = y.clamp_min(1e-6)
    return y.pow(scale)


def normalize(exp_x):
    return exp_x / exp_x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def run_variant(name, x):
    sx = stable_x(x)
    if name == "torch_softmax":
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    Path("results/csv").mkdir(parents=True, exist_ok=True)

    shapes = [(8, 128, 128)]
    if args.include_large:
        shapes.append((16, 256, 256))

    rows = []
    variants = [
        "torch_softmax",
        "lut_exp_softmax",
        "pwl_exp_softmax",
        "taylor2_exp_softmax",
        "taylor3_exp_softmax",
    ]
    for heads, q_len, k_len in shapes:
        x = torch.randn(heads, q_len, k_len, device=device)
        reference = run_variant("torch_softmax", x)
        for name in variants:
            y = run_variant(name, x)
            check_finite(name, y)
            row = {
                "experiment": "softmax_approx",
                "shape": f"heads={heads},seq={q_len}x{k_len}",
                "variant": name,
                "device": device,
                "dtype": str(y.dtype).replace("torch.", ""),
                "repeat": args.repeat,
                "warmup": args.warmup,
                "estimated_weight_size_mb": 0.0,
            }
            row.update(tensor_metrics(reference, y, include_kl=True))
            row.update(benchmark(lambda: run_variant(name, x), args.repeat, args.warmup, device))
            rows.append(row)

    out_path = Path("results/csv/softmax_approx.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
