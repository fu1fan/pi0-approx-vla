#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cpu}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pi0_approx_vla_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"

conda run -n torch python pytorch_exp/exp_linear_quant.py --device "${DEVICE}" --repeat "${REPEAT}" --warmup "${WARMUP}"
conda run -n torch python pytorch_exp/exp_projector_quant.py --device "${DEVICE}" --repeat "${REPEAT}" --warmup "${WARMUP}"
conda run -n torch python pytorch_exp/exp_softmax_approx.py --device "${DEVICE}" --repeat "${REPEAT}" --warmup "${WARMUP}"
conda run -n torch python pytorch_exp/exp_gelu_approx.py --device "${DEVICE}" --repeat "${REPEAT}" --warmup "${WARMUP}"
conda run -n torch python pytorch_exp/plot_results.py
