import argparse
import argparse
from pathlib import Path

import torch

from common.pi0_bench import (
    VARIANTS,
    append_research_log,
    bar_plot,
    check_finite,
    clear_cache,
    dtype_name,
    gated_ffn_variant,
    is_oom,
    linear_variant,
    make_ffn_weights,
    make_linear_weight,
    resolve_device,
    safe_benchmark,
    set_seed,
    tensor_metrics_with_relative_l2,
    weight_size_mb,
    write_csv,
)


def repeat_for(module, seq, variant, requested):
    repeat = requested
    if "vlm_ffn" in module or seq >= 512:
        repeat = min(repeat, 5)
    if "attention" in module and seq >= 768:
        repeat = min(repeat, 8)
    if variant in {"int8_fake_quant", "int4_weight_only_fake_quant", "w4a8_fake_quant"}:
        repeat = min(repeat, 8)
    return max(3, repeat)


def linear_case_rows(module, shape, x, w, b, args, device):
    rows = []
    ref = linear_variant(x, w, b, "fp32")
    params = w.numel() + (b.numel() if b is not None else 0)
    fp32_size = weight_size_mb(params, "fp32")
    for variant in VARIANTS:
        repeat = repeat_for(module, x.shape[1], variant, args.repeat)
        try:
            y = linear_variant(x, w, b, variant)
            check_finite(f"{module}/{variant}", y)
            size_mb = weight_size_mb(params, variant)
            row = {
                "experiment": "pi0_aligned_random_quant",
                "module": module,
                "shape": shape,
                "variant": variant,
                "device": device,
                "dtype": dtype_name(y),
                "repeat": repeat,
                "warmup": args.warmup,
                "num_parameters": params,
                "estimated_weight_size_MB": size_mb,
                "fp32_weight_size_MB": fp32_size,
                "fp16_weight_size_MB": weight_size_mb(params, "fp16"),
                "int8_weight_size_MB": weight_size_mb(params, "int8_fake_quant"),
                "int4_weight_size_MB": weight_size_mb(params, "int4_weight_only_fake_quant"),
                "compression_ratio_vs_fp32": fp32_size / size_mb,
            }
            row.update(tensor_metrics_with_relative_l2(ref, y))
            row.update(safe_benchmark(lambda v=variant: linear_variant(x, w, b, v), repeat, args.warmup, device))
            rows.append(row)
            del y
            clear_cache(device)
        except RuntimeError as exc:
            clear_cache(device)
            if is_oom(exc):
                append_research_log(f"[pi0_aligned_random_quant] OOM: {module} {shape} {variant}; skipped.")
                continue
            raise
    del ref
    return rows


def ffn_case_rows(module, shape, x, weights, args, device):
    rows = []
    gate_w, up_w, down_w, gate_b, up_b, down_b = weights
    ref = gated_ffn_variant(x, *weights, "fp32")
    params = sum(t.numel() for t in weights)
    fp32_size = weight_size_mb(params, "fp32")
    for variant in VARIANTS:
        repeat = repeat_for(module, x.shape[1], variant, args.repeat)
        try:
            y = gated_ffn_variant(x, gate_w, up_w, down_w, gate_b, up_b, down_b, variant)
            check_finite(f"{module}/{variant}", y)
            size_mb = weight_size_mb(params, variant)
            row = {
                "experiment": "pi0_aligned_random_quant",
                "module": module,
                "shape": shape,
                "variant": variant,
                "device": device,
                "dtype": dtype_name(y),
                "repeat": repeat,
                "warmup": args.warmup,
                "num_parameters": params,
                "estimated_weight_size_MB": size_mb,
                "fp32_weight_size_MB": fp32_size,
                "fp16_weight_size_MB": weight_size_mb(params, "fp16"),
                "int8_weight_size_MB": weight_size_mb(params, "int8_fake_quant"),
                "int4_weight_size_MB": weight_size_mb(params, "int4_weight_only_fake_quant"),
                "compression_ratio_vs_fp32": fp32_size / size_mb,
            }
            row.update(tensor_metrics_with_relative_l2(ref, y))
            row.update(safe_benchmark(lambda v=variant: gated_ffn_variant(x, gate_w, up_w, down_w, gate_b, up_b, down_b, v), repeat, args.warmup, device))
            rows.append(row)
            del y
            clear_cache(device)
        except RuntimeError as exc:
            clear_cache(device)
            if is_oom(exc):
                append_research_log(f"[pi0_aligned_random_quant] OOM: {module} {shape} {variant}; skipped.")
                continue
            raise
    del ref
    return rows


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    append_research_log(f"[pi0_aligned_random_quant] start device={device}, repeat={args.repeat}, warmup={args.warmup}, seed={args.seed}.")
    rows = []

    try:
        x = torch.randn(1, 768, 1152, device=device)
        w, b = make_linear_weight(2048, 1152, device)
        rows.extend(linear_case_rows("visual_projector", "[1,768,1152]->[1,768,2048]", x, w, b, args, device))
        del x, w, b
        clear_cache(device)
    except RuntimeError as exc:
        if is_oom(exc):
            append_research_log("[pi0_aligned_random_quant] OOM: visual_projector skipped.")
        else:
            raise

    attn_specs = [
        ("q_proj", 2048, 2048),
        ("k_proj", 2048, 256),
        ("v_proj", 2048, 256),
        ("o_proj", 2048, 2048),
    ]
    for seq in [256, 768, 1024]:
        x = torch.randn(1, seq, 2048, device=device)
        for name, in_dim, out_dim in attn_specs:
            w, b = make_linear_weight(out_dim, in_dim, device)
            rows.extend(linear_case_rows(f"vlm_attention_{name}_seq{seq}", f"[1,{seq},{in_dim}]->[1,{seq},{out_dim}]", x, w, b, args, device))
            del w, b
            clear_cache(device)
        del x
        clear_cache(device)

    for seq in [128, 256, 512]:
        try:
            x = torch.randn(1, seq, 2048, device=device)
            weights = make_ffn_weights(2048, 16384, 2048, device)
            rows.extend(ffn_case_rows(f"vlm_gated_ffn_seq{seq}", f"[1,{seq},2048]->[1,{seq},2048]", x, weights, args, device))
            del x, weights
            clear_cache(device)
        except RuntimeError as exc:
            clear_cache(device)
            if is_oom(exc):
                append_research_log(f"[pi0_aligned_random_quant] OOM: vlm_gated_ffn_seq{seq}; skipped.")
                continue
            raise

    x = torch.randn(1, 50, 1024, device=device)
    weights = make_ffn_weights(1024, 4096, 1024, device)
    rows.extend(ffn_case_rows("action_expert_gated_ffn", "[1,50,1024]->[1,50,1024]", x, weights, args, device))
    del x, weights
    clear_cache(device)

    proj_specs = [
        ("state_proj", 1, 32, 1024),
        ("action_in_proj", 50, 32, 1024),
        ("action_out_proj", 50, 1024, 32),
    ]
    for name, seq, in_dim, out_dim in proj_specs:
        x = torch.randn(1, seq, in_dim, device=device)
        w, b = make_linear_weight(out_dim, in_dim, device)
        rows.extend(linear_case_rows(name, f"[1,{seq},{in_dim}]->[1,{seq},{out_dim}]", x, w, b, args, device))
        del x, w, b
        clear_cache(device)

    out_csv = Path("results/csv/pi0_aligned_random_quant.csv")
    write_csv(out_csv, rows)
    bar_plot(rows, "latency_mean_ms", Path("results/figures/pi0_aligned_random_quant_latency.png"), "pi0-aligned random quant latency")
    bar_plot([r for r in rows if r["variant"] != "fp32"], "relative_l2_error", Path("results/figures/pi0_aligned_random_quant_error.png"), "pi0-aligned random quant relative L2 error", log_y=True)
    bar_plot(rows, "estimated_weight_size_MB", Path("results/figures/pi0_aligned_random_quant_size.png"), "pi0-aligned random quant weight size")

    Path("results").mkdir(exist_ok=True)
    summary = Path("results/pi0_aligned_random_quant_summary.md")
    int8 = [r for r in rows if r["variant"] == "int8_fake_quant"]
    int4 = [r for r in rows if r["variant"] == "int4_weight_only_fake_quant"]
    w4a8 = [r for r in rows if r["variant"] == "w4a8_fake_quant"]
    summary.write_text(
        "\n".join(
            [
                "# pi0-aligned Random Quantization Summary",
                "",
                "This is a random tensor proxy benchmark. It aligns shapes and module structure with key pi0/openpi-style modules, but it does not use real pi0 weights, real activations, or end-to-end robot evaluation.",
                "",
                f"- device: `{device}`",
                f"- rows: {len(rows)}",
                "- modules: visual projector, Gemma-style q/k/v/o projections, gated VLM FFN, action expert FFN, state/action projections.",
                f"- INT8 minimum cosine: {min(r['cosine_similarity'] for r in int8):.6f}" if int8 else "- INT8 rows missing.",
                f"- INT4 minimum cosine: {min(r['cosine_similarity'] for r in int4):.6f}" if int4 else "- INT4 rows missing.",
                f"- W4A8 implemented rows: {len(w4a8)}. W4A8 uses INT4 per-output-channel fake-quantized weights plus INT8 fake-quantized activations.",
                "- Interpretation: results indicate module-level sensitivity trends only. They do not represent true pi0 task success rate or full-model quantization quality.",
                "",
                "Outputs:",
                "- `results/csv/pi0_aligned_random_quant.csv`",
                "- `results/figures/pi0_aligned_random_quant_latency.png`",
                "- `results/figures/pi0_aligned_random_quant_error.png`",
                "- `results/figures/pi0_aligned_random_quant_size.png`",
            ]
        )
    )
    print(f"wrote {summary}")
    append_research_log(f"[pi0_aligned_random_quant] completed rows={len(rows)}, csv={out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
