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
    linear_variant,
    resolve_device,
    safe_benchmark,
    set_seed,
    tensor_metrics_with_relative_l2,
    weight_size_mb,
    write_csv,
)


K = {
    "visual_w": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
    "visual_b": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
    "vlm0_q": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    "vlm0_k": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
    "vlm0_v": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.v_proj.weight",
    "vlm0_o": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.o_proj.weight",
    "vlm9_q": "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.q_proj.weight",
    "vlm9_k": "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.k_proj.weight",
    "vlm9_v": "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.v_proj.weight",
    "vlm9_o": "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.o_proj.weight",
    "vlm0_gate": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
    "vlm0_up": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight",
    "vlm0_down": "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    "act_gate": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight",
    "act_up": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight",
    "act_down": "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    "state_w": "state_proj.weight",
    "state_b": "state_proj.bias",
    "action_in_w": "action_in_proj.weight",
    "action_in_b": "action_in_proj.bias",
    "action_out_w": "action_out_proj.weight",
    "action_out_b": "action_out_proj.bias",
}


def load_selected(path):
    return torch.load(path, map_location="cpu")


def tensor(data, key, device):
    return data[key].float().to(device)


def repeat_for(module, variant, requested):
    repeat = requested
    if "ffn" in module:
        repeat = min(repeat, 3)
    if variant in {"int8_fake_quant", "int4_weight_only_fake_quant", "w4a8_fake_quant"}:
        repeat = min(repeat, 5)
    return max(2, repeat)


def linear_rows(module, shape, x, w, b, args, device, original_key):
    rows = []
    ref = linear_variant(x, w, b, "fp32")
    params = w.numel() + (b.numel() if b is not None else 0)
    fp32_size = weight_size_mb(params, "fp32")
    for variant in VARIANTS:
        repeat = repeat_for(module, variant, args.repeat)
        y = linear_variant(x, w, b, variant)
        check_finite(f"{module}/{variant}", y)
        size_mb = weight_size_mb(params, variant)
        row = {
            "experiment": "pi0_real_weight_quant",
            "module": module,
            "shape": shape,
            "variant": variant,
            "device": device,
            "dtype": dtype_name(y),
            "repeat": repeat,
            "warmup": args.warmup,
            "uses_real_weights": True,
            "input_type": "random_tensor",
            "original_weight_key": original_key,
            "original_weight_shape": list(w.shape),
            "used_weight_shape": list(w.shape),
            "transposed": False,
            "num_parameters": params,
            "estimated_weight_size_MB": size_mb,
            "compression_ratio_vs_fp32": fp32_size / size_mb,
        }
        row.update(tensor_metrics_with_relative_l2(ref, y))
        row.update(safe_benchmark(lambda v=variant: linear_variant(x, w, b, v), repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    return rows


def ffn_rows(module, shape, x, weights, args, device, key_prefix):
    rows = []
    gate_w, up_w, down_w = weights
    none = None
    ref = gated_ffn_variant(x, gate_w, up_w, down_w, none, none, none, "fp32")
    params = gate_w.numel() + up_w.numel() + down_w.numel()
    fp32_size = weight_size_mb(params, "fp32")
    for variant in VARIANTS:
        repeat = repeat_for(module, variant, args.repeat)
        y = gated_ffn_variant(x, gate_w, up_w, down_w, none, none, none, variant)
        check_finite(f"{module}/{variant}", y)
        size_mb = weight_size_mb(params, variant)
        row = {
            "experiment": "pi0_real_weight_quant",
            "module": module,
            "shape": shape,
            "variant": variant,
            "device": device,
            "dtype": dtype_name(y),
            "repeat": repeat,
            "warmup": args.warmup,
            "uses_real_weights": True,
            "input_type": "random_tensor",
            "original_weight_key": key_prefix,
            "original_weight_shape": {"gate": list(gate_w.shape), "up": list(up_w.shape), "down": list(down_w.shape)},
            "used_weight_shape": {"gate": list(gate_w.shape), "up": list(up_w.shape), "down": list(down_w.shape)},
            "transposed": False,
            "num_parameters": params,
            "estimated_weight_size_MB": size_mb,
            "compression_ratio_vs_fp32": fp32_size / size_mb,
        }
        row.update(tensor_metrics_with_relative_l2(ref, y))
        row.update(safe_benchmark(lambda v=variant: gated_ffn_variant(x, gate_w, up_w, down_w, none, none, none, v), repeat, args.warmup, device))
        rows.append(row)
        del y
        clear_cache(device)
    return rows


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    data_path = Path(args.selected)
    if not data_path.exists():
        raise FileNotFoundError(f"missing selected real-weight tensor file: {data_path}")
    data = load_selected(data_path)
    append_research_log(f"[pi0_real_weight_quant] start device={device}, selected={data_path}.")
    rows = []

    x = torch.randn(1, 768, 1152, device=device)
    rows.extend(linear_rows("visual_projector", "[1,768,1152]->[1,768,2048]", x, tensor(data, K["visual_w"], device), tensor(data, K["visual_b"], device), args, device, K["visual_w"]))
    del x
    clear_cache(device)

    for layer in [0, 9]:
        prefix = f"vlm{layer}"
        x = torch.randn(1, args.attn_seq, 2048, device=device)
        for proj, out_dim in [("q", 2048), ("k", 256), ("v", 256), ("o", 2048)]:
            w = tensor(data, K[f"{prefix}_{proj}"], device)
            rows.extend(linear_rows(f"vlm_layer{layer}_{proj}_proj", f"[1,{args.attn_seq},2048]->[1,{args.attn_seq},{out_dim}]", x, w, None, args, device, K[f"{prefix}_{proj}"]))
            del w
            clear_cache(device)
        del x
        clear_cache(device)

    x = torch.randn(1, args.ffn_seq, 2048, device=device)
    vlm_weights = [tensor(data, K["vlm0_gate"], device), tensor(data, K["vlm0_up"], device), tensor(data, K["vlm0_down"], device)]
    rows.extend(ffn_rows("vlm_layer0_gated_ffn", f"[1,{args.ffn_seq},2048]->[1,{args.ffn_seq},2048]", x, vlm_weights, args, device, "paligemma language_model layer0 mlp"))
    del x, vlm_weights
    clear_cache(device)

    x = torch.randn(1, 50, 1024, device=device)
    act_weights = [tensor(data, K["act_gate"], device), tensor(data, K["act_up"], device), tensor(data, K["act_down"], device)]
    rows.extend(ffn_rows("action_expert_layer0_gated_ffn", "[1,50,1024]->[1,50,1024]", x, act_weights, args, device, "gemma_expert layer0 mlp"))
    del x, act_weights
    clear_cache(device)

    for module, seq, in_dim, out_dim, wk, bk in [
        ("state_proj", 1, 32, 1024, "state_w", "state_b"),
        ("action_in_proj", 50, 32, 1024, "action_in_w", "action_in_b"),
        ("action_out_proj", 50, 1024, 32, "action_out_w", "action_out_b"),
    ]:
        x = torch.randn(1, seq, in_dim, device=device)
        rows.extend(linear_rows(module, f"[1,{seq},{in_dim}]->[1,{seq},{out_dim}]", x, tensor(data, K[wk], device), tensor(data, K[bk], device), args, device, K[wk]))
        del x
        clear_cache(device)

    out_csv = Path("results/csv/pi0_real_weight_quant.csv")
    write_csv(out_csv, rows)
    bar_plot(rows, "latency_mean_ms", Path("results/figures/pi0_real_weight_quant_latency.png"), "pi0 real-weight quant latency")
    bar_plot([r for r in rows if r["variant"] != "fp32"], "relative_l2_error", Path("results/figures/pi0_real_weight_quant_error.png"), "pi0 real-weight quant relative L2 error", log_y=True)
    bar_plot(rows, "estimated_weight_size_MB", Path("results/figures/pi0_real_weight_quant_size.png"), "pi0 real-weight quant weight size")

    int8 = [r for r in rows if r["variant"] == "int8_fake_quant"]
    int4 = [r for r in rows if r["variant"] == "int4_weight_only_fake_quant"]
    summary = Path("results/pi0_real_weight_quant_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0 Real-weight Quantization Summary",
                "",
                "This stage uses real pi0 module weights from `lerobot/pi0_base` with random tensor inputs. It reflects quantization error under real parameter distributions, but does not represent real robot task success rate.",
                "",
                f"- device: `{device}`",
                f"- selected tensor file: `{data_path}`",
                f"- rows: {len(rows)}",
                f"- INT8 minimum cosine: {min(r['cosine_similarity'] for r in int8):.6f}" if int8 else "- INT8 rows missing.",
                f"- INT4 minimum cosine: {min(r['cosine_similarity'] for r in int4):.6f}" if int4 else "- INT4 rows missing.",
                "- All selected weights were already in `[out_dim, in_dim]` layout for `torch.nn.functional.linear`; no transpose was used.",
                "",
                "Outputs:",
                "- `results/csv/pi0_real_weight_quant.csv`",
                "- `results/figures/pi0_real_weight_quant_latency.png`",
                "- `results/figures/pi0_real_weight_quant_error.png`",
                "- `results/figures/pi0_real_weight_quant_size.png`",
            ]
        )
    )
    print(f"wrote {summary}")
    append_research_log(f"[pi0_real_weight_quant] completed rows={len(rows)}, csv={out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="results/pi0_module_weights/selected_modules.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--attn-seq", type=int, default=256)
    parser.add_argument("--ffn-seq", type=int, default=64)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
