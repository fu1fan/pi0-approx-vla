import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from common.metrics import tensor_metrics
from common.quant_utils import resolve_device
from common.timer import benchmark


def pwl_gelu(x):
    xp = torch.tensor([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0], device=x.device)
    yp = F.gelu(xp)
    xc = x.clamp(-5.0, 5.0)
    y = torch.empty_like(xc)
    for i in range(len(xp) - 1):
        lo, hi = xp[i], xp[i + 1]
        mask = (xc >= lo) & (xc <= hi)
        slope = (yp[i + 1] - yp[i]) / (hi - lo)
        y[mask] = yp[i] + slope * (xc[mask] - lo)
    y[x < -5.0] = 0.0
    y[x > 5.0] = x[x > 5.0]
    return y


def lut_gelu(x, x_min=-5.0, x_max=5.0, entries=512):
    table_x = torch.linspace(x_min, x_max, entries, device=x.device)
    table_y = F.gelu(table_x)
    pos = (x.clamp(x_min, x_max) - x_min) / (x_max - x_min) * (entries - 1)
    idx0 = torch.floor(pos).long().clamp(0, entries - 1)
    idx1 = (idx0 + 1).clamp(0, entries - 1)
    frac = pos - idx0.float()
    return table_y[idx0] * (1.0 - frac) + table_y[idx1] * frac


def run_variant(name, x):
    if name == "torch_gelu":
        return F.gelu(x)
    if name == "tanh_gelu":
        return F.gelu(x, approximate="tanh")
    if name == "pwl_gelu":
        return pwl_gelu(x)
    if name == "lut_gelu":
        return lut_gelu(x)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    Path("results/csv").mkdir(parents=True, exist_ok=True)

    shape = (1, 256, 2048)
    x = torch.empty(shape, device=device).uniform_(-5.0, 5.0)
    reference = run_variant("torch_gelu", x)
    rows = []
    for name in ["torch_gelu", "tanh_gelu", "pwl_gelu", "lut_gelu"]:
        y = run_variant(name, x)
        row = {
            "experiment": "gelu_approx",
            "shape": "1x256x2048",
            "variant": name,
            "device": device,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "estimated_weight_size_mb": 0.0,
        }
        row.update(tensor_metrics(reference, y))
        row.update(benchmark(lambda: run_variant(name, x), args.repeat, args.warmup, device))
        rows.append(row)

    out_path = Path("results/csv/gelu_approx.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
