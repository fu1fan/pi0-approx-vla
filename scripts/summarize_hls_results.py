#!/usr/bin/env python3
"""Write a markdown summary from the HLS CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISPLAY_FIELDS = [
    "kernel",
    "variant",
    "comparison_group",
    "role",
    "data_type",
    "shape",
    "latency_cycles",
    "ii",
    "estimated_clock_ns",
    "LUT",
    "FF",
    "BRAM",
    "DSP",
    "mse",
    "mae",
    "kl",
    "cosine",
    "relative_l2",
    "hls_synth_status",
    "parse_status",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str) -> str:
    return value if value not in {"", None} else "NA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/csv/hls_kernel_summary.csv", type=Path)
    parser.add_argument("--out", default="results/hls_kernel_summary.md", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    lines = [
        "# Vitis HLS Kernel Benchmark Summary",
        "",
        "This report is module-level only. It does not deploy full pi0 and does not run end-to-end VLA inference.",
        "",
        "PyTorch/Python experiments provide module error trends with random or real pi0 weights. HLS kernels provide tile-level implementability evidence: C-sim numerical checks, synthesis status, and report-derived latency/resource fields when synthesis succeeds.",
        "",
        "## Kernel Mapping",
        "",
        "- `int8_gemm`: projector / QKV / FFN / state-action projection GEMM tile.",
        "- `exact_softmax`: float exp softmax baseline for before/after comparison.",
        "- `lut_softmax`: attention score normalization.",
        "- `exact_gelu`: float tanh GELU baseline for before/after comparison.",
        "- `gelu_pwl`: FFN activation approximation.",
        "- `exact_rmsnorm`: float sqrt RMSNorm baseline for before/after comparison.",
        "- `rmsnorm_rsqrt`: Transformer RMSNorm reciprocal-square-root approximation.",
        "- `fixed_projector_tile`: optional fixed-point visual projector tile.",
        "",
        "Flow step reduction is an algorithm-level approximation and is intentionally not represented as an HLS kernel.",
        "",
        "## Results",
        "",
        "| " + " | ".join(DISPLAY_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(DISPLAY_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in DISPLAY_FIELDS) + " |")

    failures = [row for row in rows if row.get("hls_synth_status") not in {"passed"}]
    lines.extend(["", "## Failures Or Downgrades", ""])
    if failures:
        seen: set[str] = set()
        for row in failures:
            key = (row.get("kernel", ""), row.get("hls_synth_status", ""), row.get("parse_status", ""))
            if str(key) in seen:
                continue
            seen.add(str(key))
            lines.append(
                f"- `{row.get('kernel')}`: synthesis status `{fmt(row.get('hls_synth_status', ''))}`, parse status `{fmt(row.get('parse_status', ''))}`. {fmt(row.get('notes', ''))}"
            )
    else:
        lines.append("- No HLS synthesis failures recorded.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- CSV: `results/csv/hls_kernel_summary.csv`",
            "- Per-kernel status and report copies: `results/hls_reports/`",
            "- HLS source: `vitis_workspace/hls_src/`",
            "- Vitis Unified components: `vitis_workspace/<kernel>/`",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
