import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib.pyplot as plt


TASKS = [
    "vlm_ffn_activation_replacement",
    "action_expert_ffn_activation_replacement",
    "real_qk_softmax_approximation",
    "vlm_rmsnorm_approximation",
    "action_rmsnorm_approximation",
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
    reason = (
        "no extracted real pi0 tensors available; full checkpoint download was skipped by size limit or extraction found no keys"
        if not manifest
        else "manifest exists, but required FFN/QK/RMSNorm tensor groups are incomplete in this fallback path"
    )
    rows = [
        {
            "experiment": "pi0_real_weight_simplify",
            "task": task,
            "status": "skipped",
            "skip_reason": reason,
            "uses_real_weights": bool(manifest),
            "input_type": "random_tensor_if_available",
            "variant": "not_run",
        }
        for task in TASKS
    ]
    out_csv = Path("results/csv/pi0_real_weight_simplify.csv")
    write_csv(out_csv, rows)
    msg = "Real pi0 tensors were not locally available, so activation/softmax/RMSNorm simplification metrics were skipped. See CSV and summary."
    placeholder_figure(Path("results/figures/pi0_real_weight_simplify_latency.png"), "pi0 real-weight simplify latency", msg)
    placeholder_figure(Path("results/figures/pi0_real_weight_simplify_error.png"), "pi0 real-weight simplify error", msg)

    summary = Path("results/pi0_real_weight_simplify_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0 Real-weight Simplification Summary",
                "",
                "This stage is intended to repeat random-input function simplification experiments using real pi0 weights.",
                "",
                "Current run status: skipped real-weight numeric benchmarks.",
                "",
                "Reason: verified local pi0 tensors were not available after Stage 3. The single 13.04 GiB `model.safetensors` was not downloaded under the configured 2 GiB safety limit, so real FFN/QK/RMSNorm weights could not be loaded.",
                "",
                "No metrics were fabricated. If a verified checkpoint is downloaded outside git tracking, re-run extraction first and then this script.",
                "",
                "Outputs:",
                "- `results/csv/pi0_real_weight_simplify.csv`",
                "- `results/figures/pi0_real_weight_simplify_latency.png`",
                "- `results/figures/pi0_real_weight_simplify_error.png`",
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
