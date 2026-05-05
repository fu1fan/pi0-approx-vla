import argparse
import json
from pathlib import Path

import torch


PATTERNS = {
    "visual_projector": ["multi_modal_projector", "img/head/kernel", "projector", "visual_projection"],
    "vlm_attention": ["q_proj", "k_proj", "v_proj", "o_proj", "attn"],
    "vlm_ffn": ["gate_proj", "up_proj", "down_proj", "mlp"],
    "action_expert": ["expert", "action_expert", "gemma_expert", "llm_expert"],
    "action_projection": ["state_proj", "action_in_proj", "action_out_proj"],
    "rmsnorm": ["rmsnorm", "norm", "input_layernorm", "post_attention_layernorm"],
}


def load_safetensors(path, wanted):
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key in wanted:
            if key in keys:
                out[key] = f.get_tensor(key)
    return out


def read_keys(keys_file):
    keys = []
    path_by_key = {}
    for line in Path(keys_file).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            path_by_key[parts[1]] = parts[0]
            keys.append(parts[1])
    return keys, path_by_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="external/pi0_checkpoints/lerobot_pi0_base")
    parser.add_argument("--keys-file", default="results/pi0_checkpoint_keys.txt")
    parser.add_argument("--out-dir", default="results/pi0_module_weights")
    parser.add_argument("--report", default="results/pi0_extracted_modules.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ["# pi0 Extracted Modules", "", "This report records exact key matches. Missing entries are not fabricated.", ""]
    if not Path(args.keys_file).exists():
        report.append(f"- keys file missing: `{args.keys_file}`")
        Path(args.report).write_text("\n".join(report))
        return 1

    keys, path_by_key = read_keys(args.keys_file)
    if not keys:
        report.append("- no checkpoint keys available; extraction skipped.")
        Path(args.report).write_text("\n".join(report))
        return 1

    extracted = {}
    for group, patterns in PATTERNS.items():
        matches = [key for key in keys if any(pat.lower() in key.lower() for pat in patterns)]
        report.append(f"## {group}")
        if not matches:
            report.append("- no matching keys found")
            continue
        report.extend(f"- candidate: `{key}`" for key in matches[:50])
        loadable = [key for key in matches if str(path_by_key.get(key, "")).endswith(".safetensors")]
        if not loadable:
            report.append("- metadata-only keys found, but tensor files are not locally available; no tensor extracted.")
            continue
        by_file = {}
        for key in loadable:
            by_file.setdefault(path_by_key[key], []).append(key)
        tensors = {}
        for file_path, wanted in by_file.items():
            tensors.update(load_safetensors(file_path, wanted))
        save_path = out_dir / f"{group}.pt"
        torch.save({key: tensor for key, tensor in tensors.items()}, save_path)
        extracted[group] = {key: list(tensor.shape) for key, tensor in tensors.items()}
        report.append(f"- extracted tensors: {len(tensors)}")
        report.append(f"- saved: `{save_path}`")

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(extracted, indent=2))
    report.append("")
    report.append(f"manifest: `{manifest}`")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))
    print(f"wrote {report_path}")
    return 0 if extracted else 1


if __name__ == "__main__":
    raise SystemExit(main())
