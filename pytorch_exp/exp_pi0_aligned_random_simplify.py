import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common.pi0_bench import (
    append_research_log,
    bar_plot,
    check_finite,
    clear_cache,
    dtype_name,
    is_oom,
    make_ffn_weights,
    resolve_device,
    safe_benchmark,
    set_seed,
    tensor_metrics_with_relative_l2,
    write_csv,
)
from common.quant_utils import fake_quant_symmetric


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


def clipped_linear_gelu(x):
    return x * ((x + 3.0).clamp(0.0, 6.0) / 6.0)


def hard_swish(x):
    return x * ((x + 3.0).clamp(0.0, 6.0) / 6.0)


def activation(x, name):
    if name == "exact_gelu":
        return F.gelu(x)
    if name == "tanh_gelu":
        return F.gelu(x, approximate="tanh")
    if name == "pwl_gelu":
        return pwl_gelu(x)
    if name == "clipped_linear_gelu":
        return clipped_linear_gelu(x)
    if name == "identity":
        return x
    if name == "exact_silu":
        return F.silu(x)
    if name == "hard_swish":
        return hard_swish(x)
    raise ValueError(name)


def ffn_forward(x, weights, activation_name):
    gate_w, up_w, down_w, gate_b, up_b, down_b = weights
    gate = F.linear(x, gate_w, gate_b)
    up = F.linear(x, up_w, up_b)
    return F.linear(activation(gate, activation_name) * up, down_w, down_b)


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


def normalize(y):
    return y / y.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def softmax_variant(x, name):
    sx = stable_x(x)
    if name == "exact_softmax":
        return torch.softmax(sx, dim=-1)
    if name == "lut_softmax":
        return normalize(lut_exp(sx))
    if name == "pwl_softmax":
        return normalize(pwl_exp(sx))
    if name == "taylor2_softmax":
        return normalize(taylor_exp(sx, 2))
    if name == "taylor3_softmax":
        return normalize(taylor_exp(sx, 3))
    if name == "clipped_linear_normalize":
        return normalize((sx + 8.0).clamp_min(0.0))
    if name == "base2_softmax":
        return normalize(torch.pow(2.0, sx.clamp(-8.0, 0.0) / 0.6931471805599453))
    raise ValueError(name)


def rmsnorm_exact(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def nr_rsqrt(y, steps, eps=1e-6):
    yc = y.clamp_min(eps)
    guess = torch.rsqrt(yc).half().float()
    for _ in range(steps):
        guess = guess * (1.5 - 0.5 * yc * guess * guess)
    return guess


def pwl_rsqrt(y, eps=1e-6):
    yc = y.clamp(0.05, 8.0)
    xp = torch.tensor([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], device=y.device)
    yp = torch.rsqrt(xp)
    out = torch.empty_like(yc)
    for i in range(len(xp) - 1):
        lo, hi = xp[i], xp[i + 1]
        mask = (yc >= lo) & (yc <= hi)
        slope = (yp[i + 1] - yp[i]) / (hi - lo)
        out[mask] = yp[i] + slope * (yc[mask] - lo)
    return out.clamp_max(1.0 / eps**0.5)


def rmsnorm_variant(x, weight, name, eps=1e-6):
    if name == "exact_rmsnorm":
        return rmsnorm_exact(x, weight, eps)
    if name == "int8_fake_quant_rmsnorm":
        xq, _ = fake_quant_symmetric(x, 8)
        return rmsnorm_exact(xq, weight, eps)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    if name == "nr_rsqrt_1":
        return x * nr_rsqrt(var + eps, 1, eps) * weight
    if name == "nr_rsqrt_2":
        return x * nr_rsqrt(var + eps, 2, eps) * weight
    if name == "pwl_rsqrt":
        return x * pwl_rsqrt(var + eps, eps) * weight
    raise ValueError(name)


def rows_for_activation(module, x, weights, args, device):
    rows = []
    groups = [
        ("gelu_family", "exact_gelu", ["exact_gelu", "tanh_gelu", "pwl_gelu", "clipped_linear_gelu", "identity"]),
        ("silu_family", "exact_silu", ["exact_silu", "hard_swish"]),
    ]
    for group, baseline, variants in groups:
        ref = ffn_forward(x, weights, baseline)
        for name in variants:
            repeat = min(args.repeat, 5 if x.shape[1] >= 128 and x.shape[-1] >= 2048 else args.repeat)
            y = ffn_forward(x, weights, name)
            check_finite(f"{module}/{name}", y)
            row = {
                "experiment": "pi0_aligned_random_simplify",
                "module": module,
                "subexperiment": "ffn_activation",
                "shape": f"{list(x.shape)}",
                "variant": name,
                "baseline": baseline,
                "family": group,
                "device": device,
                "dtype": dtype_name(y),
                "repeat": repeat,
                "warmup": args.warmup,
                "kl_divergence": "not_applicable",
            }
            row.update(tensor_metrics_with_relative_l2(ref, y))
            row.update(safe_benchmark(lambda v=name: ffn_forward(x, weights, v), repeat, args.warmup, device))
            rows.append(row)
            del y
            clear_cache(device)
        del ref
    return rows


def rows_for_softmax(module, heads, q_len, k_len, args, device):
    rows = []
    x = torch.randn(heads, q_len, k_len, device=device)
    variants = ["exact_softmax", "lut_softmax", "pwl_softmax", "taylor2_softmax", "taylor3_softmax", "clipped_linear_normalize", "base2_softmax"]
    ref = softmax_variant(x, "exact_softmax")
    repeat = min(args.repeat, 5 if q_len * k_len * heads >= 8 * 1024 * 1024 else args.repeat)
    for name in variants:
        y = softmax_variant(x, name)
        check_finite(f"{module}/{name}", y)
        row = {
            "experiment": "pi0_aligned_random_simplify",
            "module": module,
            "subexperiment": "softmax",
            "shape": f"heads={heads},q={q_len},k={k_len}",
            "variant": name,
            "baseline": "exact_softmax",
            "family": "softmax",
            "device": device,
            "dtype": dtype_name(y),
            "repeat": repeat,
            "warmup": args.warmup,
        }
        row.update(tensor_metrics_with_relative_l2(ref, y, include_kl=True))
        row.update(safe_benchmark(lambda v=name: softmax_variant(x, v), repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    del x, ref
    return rows


def rows_for_rmsnorm(module, shape, args, device):
    rows = []
    x = torch.randn(*shape, device=device)
    weight = torch.empty(shape[-1], device=device).uniform_(0.9, 1.1)
    variants = ["exact_rmsnorm", "int8_fake_quant_rmsnorm", "nr_rsqrt_1", "nr_rsqrt_2", "pwl_rsqrt"]
    ref = rmsnorm_variant(x, weight, "exact_rmsnorm")
    for name in variants:
        y = rmsnorm_variant(x, weight, name)
        check_finite(f"{module}/{name}", y)
        row = {
            "experiment": "pi0_aligned_random_simplify",
            "module": module,
            "subexperiment": "rmsnorm",
            "shape": f"{list(shape)}",
            "variant": name,
            "baseline": "exact_rmsnorm",
            "family": "rmsnorm",
            "device": device,
            "dtype": dtype_name(y),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "kl_divergence": "not_applicable",
        }
        row.update(tensor_metrics_with_relative_l2(ref, y))
        row.update(safe_benchmark(lambda v=name: rmsnorm_variant(x, weight, v), args.repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    return rows


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    append_research_log(f"[pi0_aligned_random_simplify] start device={device}, repeat={args.repeat}, warmup={args.warmup}, seed={args.seed}.")
    rows = []

    for seq in [128, 256]:
        try:
            x = torch.randn(1, seq, 2048, device=device)
            weights = make_ffn_weights(2048, 16384, 2048, device)
            rows.extend(rows_for_activation(f"vlm_ffn_seq{seq}", x, weights, args, device))
            del x, weights
            clear_cache(device)
        except RuntimeError as exc:
            clear_cache(device)
            if is_oom(exc):
                append_research_log(f"[pi0_aligned_random_simplify] OOM: vlm_ffn_seq{seq}; skipped.")
                continue
            raise

    x = torch.randn(1, 50, 1024, device=device)
    weights = make_ffn_weights(1024, 4096, 1024, device)
    rows.extend(rows_for_activation("action_expert_ffn_seq50", x, weights, args, device))
    del x, weights
    clear_cache(device)

    for q_len, k_len in [(768, 768), (1024, 1024)]:
        try:
            rows.extend(rows_for_softmax(f"vlm_self_attention_q{q_len}_k{k_len}", 8, q_len, k_len, args, device))
        except RuntimeError as exc:
            clear_cache(device)
            if is_oom(exc):
                append_research_log(f"[pi0_aligned_random_simplify] OOM: softmax q={q_len},k={k_len}; skipped.")
                continue
            raise
    for k_len in [768, 1024]:
        rows.extend(rows_for_softmax(f"action_to_context_attention_q50_k{k_len}", 8, 50, k_len, args, device))

    rows.extend(rows_for_rmsnorm("vlm_rmsnorm", (1, 768, 2048), args, device))
    rows.extend(rows_for_rmsnorm("action_rmsnorm", (1, 50, 1024), args, device))

    out_csv = Path("results/csv/pi0_aligned_random_simplify.csv")
    write_csv(out_csv, rows)
    bar_plot(rows, "latency_mean_ms", Path("results/figures/pi0_aligned_random_simplify_latency.png"), "pi0-aligned random simplify latency")
    bar_plot([r for r in rows if r["variant"] != r["baseline"]], "relative_l2_error", Path("results/figures/pi0_aligned_random_simplify_error.png"), "pi0-aligned random simplify relative L2 error", log_y=True)

    approx = [r for r in rows if r["variant"] != r["baseline"]]
    summary = Path("results/pi0_aligned_random_simplify_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0-aligned Random Simplification Summary",
                "",
                "This is a random tensor proxy benchmark using pi0-aligned shapes. It tests function replacements inside FFN, attention softmax, and RMSNorm proxy modules. It does not use real pi0 weights or real robot activations.",
                "",
                f"- device: `{device}`",
                f"- rows: {len(rows)}",
                f"- max relative L2 among approximations: {max(r['relative_l2_error'] for r in approx):.6f}" if approx else "- no approximation rows",
                "- GELU/SiLU replacements are evaluated on final FFN output, not just activation output.",
                "- Softmax approximations subtract max, clamp where appropriate, and include KL divergence.",
                "- RMSNorm approximations include INT8 fake quant and Newton-Raphson rsqrt variants.",
                "",
                "Outputs:",
                "- `results/csv/pi0_aligned_random_simplify.csv`",
                "- `results/figures/pi0_aligned_random_simplify_latency.png`",
                "- `results/figures/pi0_aligned_random_simplify_error.png`",
            ]
        )
    )
    print(f"wrote {summary}")
    append_research_log(f"[pi0_aligned_random_simplify] completed rows={len(rows)}, csv={out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
