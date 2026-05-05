import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common.pi0_bench import (
    append_research_log,
    bar_plot,
    check_finite,
    clear_cache,
    dtype_name,
    resolve_device,
    safe_benchmark,
    set_seed,
    tensor_metrics_with_relative_l2,
    write_csv,
)
from exp_pi0_aligned_random_simplify import activation, rmsnorm_variant, softmax_variant


K = {
    "vlm0_q": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    "vlm0_k": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
    "vlm0_gate": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
    "vlm0_up": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight",
    "vlm0_down": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    "act_q": "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
    "act_gate": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight",
    "act_up": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight",
    "act_down": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    "vlm_rms": "paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm.weight",
    "act_rms": "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight",
}


SOFTMAX_VARIANTS = [
    "exact_softmax",
    "lut_softmax",
    "pwl_softmax",
    "taylor2_softmax",
    "taylor3_softmax",
    "clipped_linear_normalize",
    "base2_softmax",
]


RMSNORM_VARIANTS = ["exact_rmsnorm", "int8_fake_quant_rmsnorm", "nr_rsqrt_1", "nr_rsqrt_2", "pwl_rsqrt"]


def load_selected(path):
    return torch.load(path, map_location="cpu")


def tensor(data, key, device):
    if key not in data:
        raise KeyError(f"required tensor key missing: {key}")
    return data[key].float().to(device)


def ffn_forward(x, gate_w, up_w, down_w, activation_name):
    gate = F.linear(x.float(), gate_w.float())
    up = F.linear(x.float(), up_w.float())
    hidden = activation(gate, activation_name) * up
    return F.linear(hidden, down_w.float())


def ffn_rows(module, x, weights, args, device, weight_group):
    rows = []
    gate_w, up_w, down_w = weights
    groups = [
        ("gelu_family", "exact_gelu", ["exact_gelu", "tanh_gelu", "pwl_gelu", "clipped_linear_gelu", "identity"]),
        ("silu_family", "exact_silu", ["exact_silu", "hard_swish"]),
    ]
    repeat = min(args.repeat, 3)
    for family, baseline, variants in groups:
        ref = ffn_forward(x, gate_w, up_w, down_w, baseline)
        for variant in variants:
            y = ffn_forward(x, gate_w, up_w, down_w, variant)
            check_finite(f"{module}/{variant}", y)
            row = {
                "experiment": "pi0_real_weight_simplify",
                "module": module,
                "subexperiment": "ffn_activation",
                "shape": f"{list(x.shape)}",
                "variant": variant,
                "baseline": baseline,
                "family": family,
                "device": device,
                "dtype": dtype_name(y),
                "repeat": repeat,
                "warmup": args.warmup,
                "uses_real_weights": True,
                "input_type": "random_tensor",
                "weight_group": weight_group,
                "kl_divergence": "not_applicable",
            }
            row.update(tensor_metrics_with_relative_l2(ref, y))
            row.update(safe_benchmark(lambda v=variant: ffn_forward(x, gate_w, up_w, down_w, v), repeat, args.warmup, device))
            rows.append(row)
            del y
            clear_cache(device)
        del ref
        clear_cache(device)
    return rows


def reshape_q(q, heads=8, head_dim=256):
    bsz, seq, _ = q.shape
    return q.view(bsz, seq, heads, head_dim).transpose(1, 2).contiguous()


def reshape_kv_single_head(k, heads=8, head_dim=256):
    bsz, seq, _ = k.shape
    return k.view(bsz, seq, 1, head_dim).transpose(1, 2).expand(bsz, heads, seq, head_dim).contiguous()


def vlm_self_attention_scores(x, q_w, k_w):
    q = reshape_q(F.linear(x.float(), q_w.float()))
    k = reshape_kv_single_head(F.linear(x.float(), k_w.float()))
    return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])


def action_to_context_scores(action_x, context_x, action_q_w, context_k_w):
    q = reshape_q(F.linear(action_x.float(), action_q_w.float()))
    k = reshape_kv_single_head(F.linear(context_x.float(), context_k_w.float()))
    return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])


def softmax_rows(module, scores, args, device, score_source):
    rows = []
    repeat = min(args.repeat, 5 if scores.numel() >= 4_000_000 else args.repeat)
    ref = softmax_variant(scores, "exact_softmax")
    for variant in SOFTMAX_VARIANTS:
        y = softmax_variant(scores, variant)
        check_finite(f"{module}/{variant}", y)
        row = {
            "experiment": "pi0_real_weight_simplify",
            "module": module,
            "subexperiment": "softmax",
            "shape": f"{list(scores.shape)}",
            "variant": variant,
            "baseline": "exact_softmax",
            "family": "softmax",
            "device": device,
            "dtype": dtype_name(y),
            "repeat": repeat,
            "warmup": args.warmup,
            "uses_real_weights": True,
            "input_type": "random_tensor",
            "score_source": score_source,
        }
        row.update(tensor_metrics_with_relative_l2(ref, y, include_kl=True))
        row.update(safe_benchmark(lambda v=variant: softmax_variant(scores, v), repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    del ref
    return rows


def rmsnorm_rows(module, shape, weight, args, device, weight_key):
    rows = []
    x = torch.randn(*shape, device=device)
    ref = rmsnorm_variant(x, weight, "exact_rmsnorm")
    repeat = min(args.repeat, 10)
    for variant in RMSNORM_VARIANTS:
        y = rmsnorm_variant(x, weight, variant)
        check_finite(f"{module}/{variant}", y)
        row = {
            "experiment": "pi0_real_weight_simplify",
            "module": module,
            "subexperiment": "rmsnorm",
            "shape": f"{list(shape)}",
            "variant": variant,
            "baseline": "exact_rmsnorm",
            "family": "rmsnorm",
            "device": device,
            "dtype": dtype_name(y),
            "repeat": repeat,
            "warmup": args.warmup,
            "uses_real_weights": True,
            "input_type": "random_tensor",
            "weight_key": weight_key,
            "kl_divergence": "not_applicable",
        }
        row.update(tensor_metrics_with_relative_l2(ref, y))
        row.update(safe_benchmark(lambda v=variant: rmsnorm_variant(x, weight, v), repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    del x, ref
    return rows


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    data_path = Path(args.selected)
    if not data_path.exists():
        raise FileNotFoundError(f"missing selected real-weight tensor file: {data_path}")
    data = load_selected(data_path)
    append_research_log(f"[pi0_real_weight_simplify] start device={device}, selected={data_path}.")
    rows = []

    x = torch.randn(1, args.vlm_ffn_seq, 2048, device=device)
    rows.extend(
        ffn_rows(
            "vlm_layer0_ffn_activation",
            x,
            [tensor(data, K["vlm0_gate"], device), tensor(data, K["vlm0_up"], device), tensor(data, K["vlm0_down"], device)],
            args,
            device,
            "paligemma language_model layer0 mlp",
        )
    )
    del x
    clear_cache(device)

    x = torch.randn(1, 50, 1024, device=device)
    rows.extend(
        ffn_rows(
            "action_expert_layer0_ffn_activation",
            x,
            [tensor(data, K["act_gate"], device), tensor(data, K["act_up"], device), tensor(data, K["act_down"], device)],
            args,
            device,
            "gemma_expert layer0 mlp",
        )
    )
    del x
    clear_cache(device)

    x = torch.randn(1, args.self_attn_seq, 2048, device=device)
    scores = vlm_self_attention_scores(x, tensor(data, K["vlm0_q"], device), tensor(data, K["vlm0_k"], device))
    rows.extend(softmax_rows("vlm_layer0_self_attention_softmax", scores, args, device, "real vlm layer0 q_proj/k_proj"))
    del x, scores
    clear_cache(device)

    if K["act_q"] in data:
        action_x = torch.randn(1, 50, 1024, device=device)
        context_x = torch.randn(1, args.context_len, 2048, device=device)
        scores = action_to_context_scores(action_x, context_x, tensor(data, K["act_q"], device), tensor(data, K["vlm0_k"], device))
        rows.extend(softmax_rows("action_to_context_attention_softmax", scores, args, device, "real action expert q_proj plus real vlm layer0 k_proj"))
        del action_x, context_x, scores
        clear_cache(device)
    else:
        append_research_log("[pi0_real_weight_simplify] skipped action-to-context softmax: action expert q_proj was not extracted.")

    rows.extend(rmsnorm_rows("vlm_layer0_rmsnorm", (1, args.context_len, 2048), tensor(data, K["vlm_rms"], device), args, device, K["vlm_rms"]))
    rows.extend(rmsnorm_rows("action_expert_layer0_rmsnorm", (1, 50, 1024), tensor(data, K["act_rms"], device), args, device, K["act_rms"]))

    out_csv = Path("results/csv/pi0_real_weight_simplify.csv")
    write_csv(out_csv, rows)
    bar_plot(rows, "latency_mean_ms", Path("results/figures/pi0_real_weight_simplify_latency.png"), "pi0 real-weight simplify latency")
    bar_plot([r for r in rows if r["variant"] != r["baseline"]], "relative_l2_error", Path("results/figures/pi0_real_weight_simplify_error.png"), "pi0 real-weight simplify relative L2 error", log_y=True)

    approx = [r for r in rows if r["variant"] != r["baseline"]]
    softmax_approx = [r for r in approx if r["subexperiment"] == "softmax"]
    rms_approx = [r for r in approx if r["subexperiment"] == "rmsnorm"]
    ffn_approx = [r for r in approx if r["subexperiment"] == "ffn_activation"]
    best_softmax = min(softmax_approx, key=lambda r: r["kl_divergence"]) if softmax_approx else None
    softmax_hardware_like = [r for r in softmax_approx if r["variant"] != "base2_softmax"]
    best_softmax_hardware_like = min(softmax_hardware_like, key=lambda r: r["kl_divergence"]) if softmax_hardware_like else None
    summary = Path("results/pi0_real_weight_simplify_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0 Real-weight Simplification Summary",
                "",
                "This stage repeats function simplification benchmarks using real `lerobot/pi0_base` module weights with random tensor inputs. It reflects real parameter distributions for FFN, Q/K score generation, and RMSNorm scale, but does not measure robot task success.",
                "",
                f"- device: `{device}`",
                f"- selected tensor file: `{data_path}`",
                f"- rows: {len(rows)}",
                f"- max FFN activation replacement relative L2: {max(r['relative_l2_error'] for r in ffn_approx):.6f}" if ffn_approx else "- FFN approximation rows missing.",
                f"- best softmax approximation by KL: `{best_softmax['variant']}` KL={best_softmax['kl_divergence']:.6e}, relative_l2={best_softmax['relative_l2_error']:.6f}" if best_softmax else "- Softmax approximation rows missing.",
                f"- best hardware-like softmax approximation excluding exact base-2 reformulation: `{best_softmax_hardware_like['variant']}` KL={best_softmax_hardware_like['kl_divergence']:.6e}, relative_l2={best_softmax_hardware_like['relative_l2_error']:.6f}" if best_softmax_hardware_like else "- Hardware-like softmax approximation rows missing.",
                f"- max RMSNorm approximation relative L2: {max(r['relative_l2_error'] for r in rms_approx):.6f}" if rms_approx else "- RMSNorm approximation rows missing.",
                "- Softmax scores were generated from real Q/K projection weights before applying exact and approximate softmax variants.",
                "- `base2_softmax` uses exact `torch.pow` in this PyTorch benchmark, so it is a mathematical reformulation baseline rather than a low-cost approximate hardware kernel.",
                "- RMSNorm uses real extracted scale weights where available.",
                "",
                "Outputs:",
                "- `results/csv/pi0_real_weight_simplify.csv`",
                "- `results/figures/pi0_real_weight_simplify_latency.png`",
                "- `results/figures/pi0_real_weight_simplify_error.png`",
            ]
        )
    )
    print(f"wrote {summary}")
    append_research_log(f"[pi0_real_weight_simplify] completed rows={len(rows)}, csv={out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="results/pi0_module_weights/selected_modules.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--vlm-ffn-seq", type=int, default=128)
    parser.add_argument("--self-attn-seq", type=int, default=768)
    parser.add_argument("--context-len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=567)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
