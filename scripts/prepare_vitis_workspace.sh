#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="${ROOT_DIR}/vitis_workspace"
SRC_DIR="${WS_DIR}/hls_src"

mkdir -p "${SRC_DIR}"

for kernel in int8_linear fixed_projector lut_softmax gelu_pwl; do
  mkdir -p "${SRC_DIR}/${kernel}"
  cp "${ROOT_DIR}/hls/${kernel}/"*.cpp "${SRC_DIR}/${kernel}/"
  cp "${ROOT_DIR}/hls/${kernel}/"*.h "${SRC_DIR}/${kernel}/"
  cp "${ROOT_DIR}/hls/${kernel}/"run_hls.tcl "${SRC_DIR}/${kernel}/"
  cp "${ROOT_DIR}/hls/${kernel}/"README.md "${SRC_DIR}/${kernel}/"
done

echo "Prepared Vitis workspace source copies under ${SRC_DIR}"
echo "Open Vitis with: vitis -w ${WS_DIR}"
