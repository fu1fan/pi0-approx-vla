import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib.pyplot as plt


MODULES = [
    "visual_projector",
    "vlm_attention_q_proj",
    "vlm_attention_k_proj",
    "vlm_attention_v_proj",
    "vlm_attention_o_proj",
    "vlm_ffn",
    "action_expert_ffn",
    "state_proj",
    "action_in_proj",
    "action_out_proj",
]


def load_manifest(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def placeholder_figure(path, title, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.02, 0.55, message, fontsize=11, va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"wrote {path}")


def run(args):
    manifest = load_manifest(args.manifest)
    rows = []
    if not manifest:
        reason = "no extracted real pi0 tensors available; full checkpoint download was skipped by size limit or extraction found no keys"
        for module in MODULES:
            rows.append(
                {
                    "experiment": "pi0_real_weight_quant",
                    "module": module,
                    "status": "skipped",
                    "skip_reason": reason,
                    "uses_real_weights": False,
                    "input_type": "random_tensor_if_available",
                    "original_weight_shape": "missing",
                    "used_weight_shape": "missing",
                    "transposed": "not_applicable",
                    "variant": "not_run",
                }
            )
    else:
        reason = "manifest exists, but module-specific tensor execution is not implemented in this lightweight fallback path"
        for module, tensors in manifest.items():
            rows.append(
                {
                    "experiment": "pi0_real_weight_quant",
                    "module": module,
                    "status": "skipped",
                    "skip_reason": reason,
                    "uses_real_weights": True,
                    "input_type": "random_tensor",
                    "original_weight_shape": str(tensors),
                    "used_weight_shape": "not_run",
                    "transposed": "not_run",
                    "variant": "not_run",
                }
            )

    out_csv = Path("results/csv/pi0_real_weight_quant.csv")
    write_csv(out_csv, rows)
    msg = "Real pi0 tensor weights were not locally available, so no latency/error bars were generated. See CSV and summary for skip reasons."
    placeholder_figure(Path("results/figures/pi0_real_weight_quant_latency.png"), "pi0 real-weight quant latency", msg)
    placeholder_figure(Path("results/figures/pi0_real_weight_quant_error.png"), "pi0 real-weight quant error", msg)
    placeholder_figure(Path("results/figures/pi0_real_weight_quant_size.png"), "pi0 real-weight quant size", msg)

    summary = Path("results/pi0_real_weight_quant_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0 Real-weight Quantization Summary",
                "",
                "This stage is intended to use real pi0 module weights with random tensor inputs. It can reflect module quantization error under real parameter distributions, but still cannot represent robot task success rate.",
                "",
                "Current run status: skipped real-weight numeric benchmarks.",
                "",
                "Reason: full `lerobot/pi0_base` weights are a single 13.04 GiB `model.safetensors`; the download was skipped by the 2 GiB safety limit, so no local tensor keys/modules were available for extraction.",
                "",
                "No metrics were fabricated. Re-run Stage 3 with a larger explicit download limit or a local verified checkpoint to enable this stage.",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/pi0_module_weights/manifest.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
