#!/usr/bin/env python3
"""Create before/after HLS optimization comparison CSV and markdown."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PAIRS = [
    {
        "comparison_group": "softmax",
        "baseline_kernel": "exact_softmax",
        "baseline_variant": "exp",
        "optimized_kernel": "lut_softmax",
        "optimized_variant": "default",
        "data_type": "float32 -> fixed16x6_prob18x2",
        "shape": "rows4_len128",
    },
    {
        "comparison_group": "gelu",
        "baseline_kernel": "exact_gelu",
        "baseline_variant": "tanh",
        "optimized_kernel": "gelu_pwl",
        "optimized_variant": "default",
        "data_type": "float32 -> fixed16x6",
        "shape": "len4096",
    },
    {
        "comparison_group": "rmsnorm_nr1",
        "baseline_kernel": "exact_rmsnorm",
        "baseline_variant": "sqrt",
        "optimized_kernel": "rmsnorm_rsqrt",
        "optimized_variant": "nr1",
        "data_type": "float32 -> fixed16x6_acc40x16",
        "shape": "hidden1024",
    },
    {
        "comparison_group": "rmsnorm_nr2",
        "baseline_kernel": "exact_rmsnorm",
        "baseline_variant": "sqrt",
        "optimized_kernel": "rmsnorm_rsqrt",
        "optimized_variant": "nr2",
        "data_type": "float32 -> fixed16x6_acc40x16",
        "shape": "hidden1024",
    },
]


FIELDS = [
    "comparison_group",
    "baseline_kernel",
    "optimized_kernel",
    "data_type",
    "shape",
    "baseline_latency_cycles",
    "optimized_latency_cycles",
    "latency_speedup",
    "baseline_ii",
    "optimized_ii",
    "baseline_clock_ns",
    "optimized_clock_ns",
    "baseline_estimated_time_ns",
    "optimized_estimated_time_ns",
    "time_speedup",
    "baseline_fmax_mhz",
    "optimized_fmax_mhz",
    "baseline_LUT",
    "optimized_LUT",
    "LUT_delta_pct",
    "baseline_FF",
    "optimized_FF",
    "FF_delta_pct",
    "baseline_BRAM",
    "optimized_BRAM",
    "BRAM_delta_pct",
    "baseline_DSP",
    "optimized_DSP",
    "DSP_delta_pct",
    "MSE",
    "MAE",
    "KL",
    "cosine",
    "relative_l2",
    "baseline_MSE",
    "baseline_cosine",
    "optimized_variant",
    "status",
    "notes",
]


DISPLAY_FIELDS = [
    "comparison_group",
    "baseline_kernel",
    "optimized_kernel",
    "shape",
    "baseline_latency_cycles",
    "optimized_latency_cycles",
    "latency_speedup",
    "baseline_clock_ns",
    "optimized_clock_ns",
    "baseline_estimated_time_ns",
    "optimized_estimated_time_ns",
    "time_speedup",
    "baseline_LUT",
    "optimized_LUT",
    "LUT_delta_pct",
    "baseline_DSP",
    "optimized_DSP",
    "DSP_delta_pct",
    "MSE",
    "cosine",
    "status",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("kernel", ""), row.get("variant", "default") or "default"


def as_float(value: str) -> float | None:
    if value in {"", "NA", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.6e}"
    return f"{value:.6f}"


def pct_delta(baseline: str, optimized: str) -> str:
    b = as_float(baseline)
    o = as_float(optimized)
    if b is None or o is None or b == 0:
        return ""
    return fmt_num((o - b) / b * 100.0)


def speedup(baseline: str, optimized: str) -> str:
    b = as_float(baseline)
    o = as_float(optimized)
    if b is None or o is None or o == 0:
        return ""
    return fmt_num(b / o)


def estimated_time_ns(latency_cycles: str, clock_ns: str) -> str:
    cycles = as_float(latency_cycles)
    clock = as_float(clock_ns)
    if cycles is None or clock is None:
        return ""
    return fmt_num(cycles * clock)


def fmax_mhz(clock_ns: str) -> str:
    clock = as_float(clock_ns)
    if clock is None or clock == 0:
        return ""
    return fmt_num(1000.0 / clock)


def status_and_notes(baseline: dict[str, str] | None, optimized: dict[str, str] | None) -> tuple[str, str]:
    notes: list[str] = []
    if baseline is None:
        notes.append("missing baseline row")
    if optimized is None:
        notes.append("missing optimized row")
    if baseline is None or optimized is None:
        return "incomplete", "; ".join(notes)

    for label, row in (("baseline", baseline), ("optimized", optimized)):
        if row.get("hls_synth_status") != "passed":
            notes.append(f"{label} synthesis {row.get('hls_synth_status', 'unknown')}")
        if row.get("parse_status") != "ok":
            notes.append(f"{label} parse {row.get('parse_status', 'unknown')}")
        if row.get("notes"):
            notes.append(f"{label}: {row['notes']}")

    return ("passed" if not notes else "partial", "; ".join(notes))


def make_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {key(row): row for row in rows}
    out: list[dict[str, str]] = []
    for pair in PAIRS:
        baseline = by_key.get((pair["baseline_kernel"], pair["baseline_variant"]))
        optimized = by_key.get((pair["optimized_kernel"], pair["optimized_variant"]))
        status, notes = status_and_notes(baseline, optimized)
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "comparison_group": pair["comparison_group"],
                "baseline_kernel": pair["baseline_kernel"],
                "optimized_kernel": pair["optimized_kernel"],
                "data_type": pair["data_type"],
                "shape": pair["shape"],
                "optimized_variant": pair["optimized_variant"],
                "status": status,
                "notes": notes,
            }
        )
        if baseline is not None:
            row.update(
                {
                    "baseline_latency_cycles": baseline.get("latency_cycles", ""),
                    "baseline_ii": baseline.get("ii", ""),
                    "baseline_clock_ns": baseline.get("estimated_clock_ns", ""),
                    "baseline_LUT": baseline.get("LUT", ""),
                    "baseline_FF": baseline.get("FF", ""),
                    "baseline_BRAM": baseline.get("BRAM", ""),
                    "baseline_DSP": baseline.get("DSP", ""),
                    "baseline_MSE": baseline.get("mse", ""),
                    "baseline_cosine": baseline.get("cosine", ""),
                }
            )
        if optimized is not None:
            row.update(
                {
                    "optimized_latency_cycles": optimized.get("latency_cycles", ""),
                    "optimized_ii": optimized.get("ii", ""),
                    "optimized_clock_ns": optimized.get("estimated_clock_ns", ""),
                    "optimized_LUT": optimized.get("LUT", ""),
                    "optimized_FF": optimized.get("FF", ""),
                    "optimized_BRAM": optimized.get("BRAM", ""),
                    "optimized_DSP": optimized.get("DSP", ""),
                    "MSE": optimized.get("mse", ""),
                    "MAE": optimized.get("mae", ""),
                    "KL": optimized.get("kl", ""),
                    "cosine": optimized.get("cosine", ""),
                    "relative_l2": optimized.get("relative_l2", ""),
                }
            )
        row["latency_speedup"] = speedup(row["baseline_latency_cycles"], row["optimized_latency_cycles"])
        row["baseline_estimated_time_ns"] = estimated_time_ns(row["baseline_latency_cycles"], row["baseline_clock_ns"])
        row["optimized_estimated_time_ns"] = estimated_time_ns(row["optimized_latency_cycles"], row["optimized_clock_ns"])
        row["time_speedup"] = speedup(row["baseline_estimated_time_ns"], row["optimized_estimated_time_ns"])
        row["baseline_fmax_mhz"] = fmax_mhz(row["baseline_clock_ns"])
        row["optimized_fmax_mhz"] = fmax_mhz(row["optimized_clock_ns"])
        row["LUT_delta_pct"] = pct_delta(row["baseline_LUT"], row["optimized_LUT"])
        row["FF_delta_pct"] = pct_delta(row["baseline_FF"], row["optimized_FF"])
        row["BRAM_delta_pct"] = pct_delta(row["baseline_BRAM"], row["optimized_BRAM"])
        row["DSP_delta_pct"] = pct_delta(row["baseline_DSP"], row["optimized_DSP"])
        out.append(row)
    return out


def fmt_cell(value: str) -> str:
    return value if value not in {"", None} else "NA"


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# HLS Optimization Before/After Comparison",
        "",
        "This report compares optimization-before exact-style HLS baselines against the existing approximate module-level kernels. It does not include power and does not represent full pi0 deployment.",
        "",
        "## Comparison Contract",
        "",
        "- Softmax: float exp row-wise softmax baseline vs LUT exp softmax.",
        "- GELU: float tanh GELU baseline vs 16-segment PWL GELU.",
        "- RMSNorm: float sqrt RMSNorm baseline vs LUT-initialized Newton-Raphson rsqrt branches.",
        "- Metrics come from HLS C-sim and C-synthesis reports when synthesis succeeds.",
        "",
        "## Results",
        "",
        "| " + " | ".join(DISPLAY_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(DISPLAY_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_cell(row.get(field, "")) for field in DISPLAY_FIELDS) + " |")

    lines.extend(["", "## Notes", ""])
    for row in rows:
        note = row.get("notes", "")
        if note:
            lines.append(f"- `{row['comparison_group']}`: {note}")
    if not any(row.get("notes") for row in rows):
        lines.append("- All comparison pairs synthesized and parsed successfully.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `latency_speedup` is `baseline_latency / optimized_latency`; values above 1 mean the optimized kernel has fewer cycles.",
            "- `time_speedup` uses `latency_cycles * estimated_clock_ns`; this captures cases where an approximate kernel has similar cycles but a better estimated clock.",
            "- Resource delta columns are `(optimized - baseline) / baseline * 100`; negative values mean the optimized kernel used fewer resources.",
            "- Error columns report the optimized kernel error versus the Python/C++ golden reference. Baseline numeric error is retained in `baseline_MSE` and `baseline_cosine` in the CSV.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/csv/hls_kernel_summary.csv", type=Path)
    parser.add_argument("--out-csv", default="results/csv/hls_optimization_comparison.csv", type=Path)
    parser.add_argument("--out-md", default="results/hls_optimization_comparison.md", type=Path)
    args = parser.parse_args()

    rows = make_comparison_rows(read_rows(args.csv))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out_md)
    print(f"wrote {len(rows)} rows to {args.out_csv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
