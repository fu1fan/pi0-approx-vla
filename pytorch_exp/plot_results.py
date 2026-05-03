import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib.pyplot as plt
import pandas as pd
import torch


CSV_DIR = Path("results/csv")
FIG_DIR = Path("results/figures")
SUMMARY_PATH = Path("results/summary.md")

EXPECTED_CSV = [
    CSV_DIR / "linear_quant.csv",
    CSV_DIR / "projector_quant.csv",
    CSV_DIR / "softmax_approx.csv",
    CSV_DIR / "gelu_rmsnorm_approx.csv",
]


def load_results():
    frames = []
    for path in EXPECTED_CSV:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_csv"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def short_label(row):
    experiment = str(row["experiment"]).replace("_", " ")
    variant = str(row["variant"]).replace("_", " ")
    shape = str(row["shape"])
    return f"{experiment}\n{variant}\n{shape}"


def plot_metric(df, metric, title, out_name):
    if metric not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    labels = [short_label(row) for _, row in plot_df.iterrows()]
    fig_width = max(8, min(18, 0.55 * len(df)))
    plt.figure(figsize=(fig_width, 5))
    plt.bar(range(len(plot_df)), plot_df[metric])
    plt.xticks(range(len(plot_df)), labels, rotation=70, ha="right", fontsize=8)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    out_path = FIG_DIR / out_name
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"wrote {out_path}")


def best_rows(df, experiment, metric="cosine_similarity", n=1):
    subset = df[df["experiment"].eq(experiment)].copy()
    if subset.empty or metric not in subset.columns:
        return subset
    subset = subset[~subset["variant"].astype(str).str.contains("fp32|fp16|torch_softmax|torch_gelu|fp32_rmsnorm")]
    return subset.sort_values(metric, ascending=False).head(n)


def env_info():
    gpu = "no cuda"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu": gpu,
    }


def maybe_nvidia_smi():
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "nvidia-smi not found"
    if proc.returncode == 0:
        return proc.stdout.strip()
    message = proc.stderr.strip() or proc.stdout.strip()
    return message.splitlines()[0] if message else "nvidia-smi failed without diagnostic output"


def summarize_top_ppt(df):
    lines = []
    for experiment in ["linear_quant", "projector_quant", "softmax_approx", "gelu_approx", "rmsnorm_approx"]:
        row_df = best_rows(df, experiment)
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        lines.append(
            f"- {row['experiment']} / {row['variant']}: cosine={row.get('cosine_similarity', float('nan')):.6f}, "
            f"MSE={row.get('mse', float('nan')):.3e}, latency={row.get('latency_mean_ms', float('nan')):.3f} ms."
        )
    return lines[:3]


def build_conclusions(df):
    lines = []
    int8 = df[df["variant"].astype(str).str.contains("int8", case=False, na=False)]
    if not int8.empty:
        good = int8[int8["cosine_similarity"] >= 0.999]
        lines.append(
            f"- INT8 is suitable for {', '.join(sorted(good['experiment'].unique())) if not good.empty else 'no module at the current 0.999 cosine threshold'}."
        )
    int4 = df[df["variant"].astype(str).str.contains("int4", case=False, na=False)]
    if not int4.empty:
        worst = int4.sort_values("mse", ascending=False).iloc[0]
        lines.append(f"- INT4 weight-only has the largest observed error on {worst['experiment']} ({worst['shape']}).")
    softmax = df[df["experiment"].eq("softmax_approx")]
    if not softmax.empty:
        approx = softmax[softmax["variant"].ne("torch_softmax")]
        if not approx.empty:
            stable = approx.sort_values(["kl_divergence", "mse"], ascending=True).iloc[0]
            lines.append(f"- Softmax approximation is most stable with {stable['variant']} by KL/MSE on the current run.")
    if torch.cuda.is_available():
        lines.append("- Current CSV and figures are from CUDA runs.")
    else:
        lines.append("- CUDA experiments should be rerun after the NVIDIA driver is visible to PyTorch; current measurements may be CPU fallback.")
    return lines


def write_summary(df):
    info = env_info()
    ppt_lines = summarize_top_ppt(df)
    conclusions = build_conclusions(df)
    completed = sorted(df["experiment"].unique()) if not df.empty else []
    toy_csv = CSV_DIR / "toy_flow_matching.csv"
    toy_fig = FIG_DIR / "toy_flow_matching_curve.png"
    if toy_csv.exists() and "toy_flow_matching" not in completed:
        completed.append("toy_flow_matching")
    csv_paths = [str(path) for path in EXPECTED_CSV if path.exists()]
    if toy_csv.exists():
        csv_paths.append(str(toy_csv))
    figure_paths = [
        "results/figures/latency_compare.png",
        "results/figures/error_compare.png",
        "results/figures/cosine_compare.png",
        "results/figures/model_size_compare.png",
    ]
    if toy_fig.exists():
        figure_paths.append(str(toy_fig))
    text = [
        "# PyTorch Experiment Summary",
        "",
        "## Environment",
        f"- Conda env: torch",
        f"- Python: {info['python']}",
        f"- PyTorch: {info['pytorch']}",
        f"- CUDA available: {info['cuda_available']}",
        f"- GPU: {info['gpu']}",
        f"- nvidia-smi: {maybe_nvidia_smi()}",
        "",
        "## Completed Experiments",
    ]
    text.extend([f"- {name}" for name in completed])
    text.extend(["", "## CSV Outputs"])
    text.extend([f"- `{path}`" for path in csv_paths])
    text.extend(["", "## Figures"])
    text.extend([f"- `{path}`" for path in figure_paths if Path(path).exists()])
    text.extend(["", "## PPT-Ready Results"])
    text.extend(ppt_lines or ["- No valid result rows found."])
    text.extend(["", "## Current Conclusions"])
    text.extend(conclusions)
    text.extend(
        [
            "",
            "## Next Steps for Vitis HLS",
            "- Prioritize INT8 linear/projector GEMM kernels first because they map directly to bandwidth and DSP savings.",
            "- Implement LUT or PWL softmax exp next; validate with KL divergence before integration.",
            "- Keep GELU LUT/PWL and RMSNorm reciprocal-sqrt kernels as smaller nonlinear accelerator targets.",
            "- Use the current CUDA CSVs as the PPT latency source; rerun only if changing repeat/shape settings.",
            "",
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(text))
    print(f"wrote {SUMMARY_PATH}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    if df.empty:
        print("no CSV files found")
        return

    plot_metric(df, "latency_mean_ms", "Latency Comparison", "latency_compare.png")
    plot_metric(df, "mse", "Error Comparison (MSE)", "error_compare.png")
    plot_metric(df, "cosine_similarity", "Cosine Similarity Comparison", "cosine_compare.png")
    size_df = df[df["estimated_weight_size_mb"] > 0].copy()
    if not size_df.empty:
        plot_metric(size_df, "estimated_weight_size_mb", "Estimated Weight Size", "model_size_compare.png")
    write_summary(df)


if __name__ == "__main__":
    main()
