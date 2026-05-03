#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "torch"; then
  echo "Conda environment 'torch' was not found. Create or restore it before installing dependencies."
  exit 1
fi

conda run -n torch python -m pip install -r "${ROOT_DIR}/requirements.txt"
