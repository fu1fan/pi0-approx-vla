import argparse
import json
from pathlib import Path

import torch


SELECTED_KEYS = {
    "visual_projector": [
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
    ],
    "vlm_attention_layer0": [
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.v_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.o_proj.weight",
    ],
    "vlm_attention_layer9": [
        "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.q_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.k_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.v_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.o_proj.weight",
    ],
    "vlm_ffn_layer0": [
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    ],
    "vlm_ffn_layer9": [
        "paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.gate_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.up_proj.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.down_proj.weight",
    ],
    "action_expert_attention_layer0": [
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.o_proj.weight",
    ],
    "action_expert_ffn_layer0": [
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    ],
    "action_projection": [
        "state_proj.weight",
        "state_proj.bias",
        "action_in_proj.weight",
        "action_in_proj.bias",
        "action_out_proj.weight",
        "action_out_proj.bias",
    ],
    "rmsnorm": [
        "paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight",
    ],
}


def read_keys(keys_file):
    keys = []
    path_by_key = {}
    shape_by_key = {}
    for line in Path(keys_file).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            path_by_key[parts[1]] = parts[0]
            shape_by_key[parts[1]] = parts[2]
            keys.append(parts[1])
    return set(keys), path_by_key, shape_by_key


def load_safetensors(path, wanted):
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        available = set(f.keys())
        for key in wanted:
            if key in available:
                out[key] = f.get_tensor(key)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="external/pi0_checkpoints/lerobot_pi0_base")
    parser.add_argument("--keys-file", default="results/pi0_checkpoint_keys.txt")
    parser.add_argument("--out-dir", default="results/pi0_module_weights")
    parser.add_argument("--report", default="results/pi0_extracted_modules.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ["# pi0 Extracted Modules", "", "This report records selected exact key matches. Missing entries are not fabricated.", ""]
    keys_file = Path(args.keys_file)
    if not keys_file.exists():
        report.append(f"- keys file missing: `{args.keys_file}`")
        Path(args.report).write_text("\n".join(report))
        return 1

    keys, path_by_key, shape_by_key = read_keys(keys_file)
    if not keys:
        report.append("- no checkpoint keys available; extraction skipped.")
        Path(args.report).write_text("\n".join(report))
        return 1

    selected = {}
    manifest = {}
    missing = {}
    for group, wanted_keys in SELECTED_KEYS.items():
        present = [key for key in wanted_keys if key in keys]
        absent = [key for key in wanted_keys if key not in keys]
        report.append(f"## {group}")
        if absent:
            report.extend(f"- missing: `{key}`" for key in absent)
            missing[group] = absent
        if not present:
            report.append("- no selected tensors found")
            continue
        by_file = {}
        for key in present:
            by_file.setdefault(path_by_key[key], []).append(key)
            report.append(f"- selected: `{key}` shape `{shape_by_key.get(key, 'unknown')}`")
        tensors = {}
        for file_path, wanted in by_file.items():
            tensors.update(load_safetensors(file_path, wanted))
        for key, tensor in tensors.items():
            selected[key] = tensor.cpu()
        manifest[group] = {key: {"shape": list(tensors[key].shape), "dtype": str(tensors[key].dtype), "source": path_by_key[key]} for key in tensors}
        report.append(f"- extracted tensors: {len(tensors)}")

    tensor_path = out_dir / "selected_modules.pt"
    torch.save(selected, tensor_path)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"groups": manifest, "missing": missing, "tensor_file": str(tensor_path)}, indent=2))
    report.append("")
    report.append(f"selected tensor file: `{tensor_path}`")
    report.append(f"manifest: `{manifest_path}`")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))
    print(f"wrote {report_path}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
