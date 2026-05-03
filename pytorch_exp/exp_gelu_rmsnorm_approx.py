import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from common.metrics import tensor_metrics
from common.quant_utils import fake_quant_symmetric, resolve_device
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


def rms_norm_fp32(x, weight, eps=1e-6):
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return x32 * torch.rsqrt(var + eps) * weight.float()


def rms_norm_fp16(x, weight, eps=1e-6):
    x16 = x.half()
    w16 = weight.half()
    var = x16.pow(2).mean(dim=-1, keepdim=True)
    return (x16 * torch.rsqrt(var + eps) * w16).float()


def rms_norm_int8_input_fake_quant(x, weight, eps=1e-6):
    xq, _ = fake_quant_symmetric(x, 8)
    return rms_norm_fp32(xq, weight, eps)


def approx_rsqrt_newton(y, eps=1e-6):
    yc = y.clamp_min(eps)
    guess = torch.rsqrt(yc).half().float()
    return guess * (1.5 - 0.5 * yc * guess * guess)


def rms_norm_approx_rsqrt(x, weight, eps=1e-6):
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    inv = approx_rsqrt_newton(var + eps)
    return x32 * inv * weight.float()


def run_gelu_variant(name, x):
    if name == "torch_gelu":
        return F.gelu(x)
    if name == "tanh_gelu":
        return F.gelu(x, approximate="tanh")
    if name == "pwl_gelu":
        return pwl_gelu(x)
    if name == "lut_gelu":
        return lut_gelu(x)
    raise ValueError(name)


def run_rmsnorm_variant(name, x, weight):
    if name == "fp32_rmsnorm":
        return rms_norm_fp32(x, weight)
    if name == "fp16_rmsnorm":
        return rms_norm_fp16(x, weight)
    if name == "int8_input_fake_quant_rmsnorm":
        return rms_norm_int8_input_fake_quant(x, weight)
    if name == "approx_rsqrt_rmsnorm":
        return rms_norm_approx_rsqrt(x, weight)
    raise ValueError(name)


def check_finite(name, y):
    if not torch.isfinite(y).all():
        raise RuntimeError(f"{name} produced NaN or Inf")


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

    shape = (1, 256, 2048)
    x = torch.empty(shape, device=device).uniform_(-5.0, 5.0)
    weight = torch.empty(shape[-1], device=device).uniform_(0.9, 1.1)
    rows = []

    gelu_reference = run_gelu_variant("torch_gelu", x)
    for name in ["torch_gelu", "tanh_gelu", "pwl_gelu", "lut_gelu"]:
        y = run_gelu_variant(name, x)
        check_finite(name, y)
        row = {
            "experiment": "gelu_approx",
            "shape": "1x256x2048",
            "variant": name,
            "device": device,
            "dtype": str(y.dtype).replace("torch.", ""),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "estimated_weight_size_mb": 0.0,
        }
        row.update(tensor_metrics(gelu_reference, y))
        row.update(benchmark(lambda: run_gelu_variant(name, x), args.repeat, args.warmup, device))
        rows.append(row)

    rms_reference = run_rmsnorm_variant("fp32_rmsnorm", x, weight)
    for name in ["fp32_rmsnorm", "fp16_rmsnorm", "int8_input_fake_quant_rmsnorm", "approx_rsqrt_rmsnorm"]:
        y = run_rmsnorm_variant(name, x, weight)
        check_finite(name, y)
        row = {
            "experiment": "rmsnorm_approx",
            "shape": "1x256x2048",
            "variant": name,
            "device": device,
            "dtype": str(y.dtype).replace("torch.", ""),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "estimated_weight_size_mb": shape[-1] * 4 / (1024 * 1024),
        }
        row.update(tensor_metrics(rms_reference, y))
        row.update(benchmark(lambda: run_rmsnorm_variant(name, x, weight), args.repeat, args.warmup, device))
        rows.append(row)

    out_path = Path("results/csv/gelu_rmsnorm_approx.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
