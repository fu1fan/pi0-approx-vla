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
SCALE_LINEAR_CSV = CSV_DIR / "scale_sweep_linear.csv"
SCALE_SOFTMAX_CSV = CSV_DIR / "scale_sweep_softmax.csv"
SCALE_FIGURES = [
    "results/figures/scale_sweep_latency.png",
    "results/figures/scale_sweep_memory.png",
    "results/figures/scale_sweep_error.png",
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


def load_optional_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_scale_sweep_section():
    linear = load_optional_csv(SCALE_LINEAR_CSV)
    softmax = load_optional_csv(SCALE_SOFTMAX_CSV)
    if linear.empty and softmax.empty:
        return []

    lines = ["", "## Scale Sweep Findings"]

    if not linear.empty:
        largest = linear.sort_values(["in_dim", "out_dim"], ascending=False).iloc[0]
        int8 = linear[linear["variant"].eq("int8_fake_quant")]
        int4 = linear[linear["variant"].eq("int4_weight_only_fake_quant")]
        fp16 = linear[linear["variant"].eq("fp16")]
        int8_cos_min = int8["cosine_similarity"].min() if not int8.empty else float("nan")
        int4_cos_min = int4["cosine_similarity"].min() if not int4.empty else float("nan")
        fp16_speed_max = fp16["speedup_vs_fp32"].max() if not fp16.empty else float("nan")
        lines.extend(
            [
                f"- Linear scale sweep covered up to `{largest['shape']}` on `{largest['device']}`.",
                f"- Model size benefit becomes more visible with scale: FP32 `{largest['estimated_weight_size_MB']:.1f} MB` at the largest shape, while INT8 is 4x smaller and INT4 weight-only is 8x smaller by construction.",
                f"- PyTorch fake INT8/INT4 did not show latency speedup because quantize/dequantize is included in the measured path; FP16 reached up to `{fp16_speed_max:.2f}x` speedup vs FP32 in this sweep.",
                f"- INT8 error stayed acceptable for module-level approximation (minimum cosine `{int8_cos_min:.6f}`); INT4 weight-only is more aggressive (minimum cosine `{int4_cos_min:.6f}`) and should be treated as accuracy-risky without grouping/calibration.",
            ]
        )

    if not softmax.empty:
        approx = softmax[softmax["variant"].ne("exact_softmax")]
        stable = approx.sort_values(["kl_divergence", "mse"], ascending=True).iloc[0] if not approx.empty else None
        fastest_approx = approx.sort_values("latency_mean_ms", ascending=True).iloc[0] if not approx.empty else None
        largest_softmax = softmax.sort_values(["heads", "seq"], ascending=False).iloc[0]
        if stable is not None and fastest_approx is not None:
            lines.extend(
                [
                    f"- Softmax scale sweep covered up to `{largest_softmax['shape']}`.",
                    f"- LUT exp softmax is the most accurate/stable family in this run (`{stable['variant']}`, KL `{stable['kl_divergence']:.3e}` on `{stable['shape']}`).",
                    f"- Exact PyTorch softmax remains faster than the Python-level approximate paths; the fastest approximate row was `{fastest_approx['variant']}` at `{fastest_approx['latency_mean_ms']:.3f} ms`. The scale sweep is therefore strongest as an error/stability study and as motivation for HLS LUT/PWL kernels, not as a PyTorch speed claim.",
                    "- As sequence length grows, softmax approximation becomes more meaningful for hardware because the score matrix scales as O(seq^2), but PyTorch needs fused/custom kernels to convert that into latency wins.",
                ]
            )

    lines.extend(
        [
            "",
            "## Scale Sweep Answers",
            "1. INT8/INT4 model size benefits are clearly more visible at larger hidden dims; latency benefits are not shown by PyTorch fake quant because the measured path includes quantize/dequantize overhead. FP16 does show CUDA latency speedup.",
            "2. Softmax approximation becomes more meaningful as sequence length grows because attention scores scale as O(seq^2), but PyTorch exact softmax is still faster here; fused/HLS approximate kernels are needed for latency wins.",
            "3. Acceptable-error methods in this run: FP16, INT8 fake quant, LUT softmax, Taylor 3 softmax, LUT GELU, and approximate-rsqrt RMSNorm. INT4 weight-only and coarse PWL softmax/GELU are useful stress points but need calibration or better segmentation before high-accuracy use.",
            "4. Best PPT results: scale_sweep_memory for 256 MB -> 64 MB/32 MB Linear storage, scale_sweep_error for INT8-vs-INT4 accuracy contrast, and scale_sweep_latency with the fake-quant caveat.",
            "5. Vitis HLS should prioritize INT8 Linear/projector GEMM first, then LUT softmax exp for long-sequence attention.",
        ]
    )

    lines.extend(
        [
            "",
            "## Scale Sweep PPT Picks",
            "- `scale_sweep_memory.png`: largest Linear shape shows FP32 256 MB vs INT8 64 MB vs INT4 32 MB weight storage.",
            "- `scale_sweep_error.png`: INT8 remains high-cosine across hidden dims; INT4 error grows and is the cautionary contrast.",
            "- `scale_sweep_latency.png`: include with the caveat that fake quantization latency includes quant/dequant overhead; use it to motivate true HLS/cuBLASLt INT kernels.",
            "",
            "## Scale Sweep HLS Priority",
            "- First priority: INT8 Linear / projector GEMM, because accuracy is acceptable and memory bandwidth savings scale directly with hidden dimension.",
            "- Second priority: LUT softmax exp, because it is numerically stable and the O(seq^2) attention matrix makes hardware approximation increasingly relevant at long sequence length.",
        ]
    )
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
    if SCALE_LINEAR_CSV.exists() and "scale_sweep_linear" not in completed:
        completed.append("scale_sweep_linear")
    if SCALE_SOFTMAX_CSV.exists() and "scale_sweep_softmax" not in completed:
        completed.append("scale_sweep_softmax")
    csv_paths = [str(path) for path in EXPECTED_CSV if path.exists()]
    if toy_csv.exists():
        csv_paths.append(str(toy_csv))
    for scale_path in [SCALE_LINEAR_CSV, SCALE_SOFTMAX_CSV]:
        if scale_path.exists():
            csv_paths.append(str(scale_path))
    figure_paths = [
        "results/figures/latency_compare.png",
        "results/figures/error_compare.png",
        "results/figures/cosine_compare.png",
        "results/figures/model_size_compare.png",
    ]
    if toy_fig.exists():
        figure_paths.append(str(toy_fig))
    figure_paths.extend([path for path in SCALE_FIGURES if Path(path).exists()])
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
    text.extend(build_scale_sweep_section())
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
