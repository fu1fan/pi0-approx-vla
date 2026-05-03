#!/usr/bin/env bash
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${ROOT_DIR}/docs/environment_check.md"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pi0_approx_vla_matplotlib}"
mkdir -p "${ROOT_DIR}/docs"
mkdir -p "${MPLCONFIGDIR}"

{
  echo "# Environment Check"
  echo
  echo "Generated at: $(date)"
  echo
  echo "## Working Directory"
  echo '```text'
  pwd
  echo '```'
  echo
  echo "## Root Listing"
  echo '```text'
  ls -la "${ROOT_DIR}"
  echo '```'
  echo
  echo "## Conda Environments"
  echo '```text'
  conda env list
  echo '```'
  echo
  echo "## Torch Conda Environment"
  echo '```text'
  if conda env list | awk '{print $1}' | grep -qx "torch"; then
    echo "torch environment: found"
  else
    echo "torch environment: missing"
  fi
  echo '```'
  echo
  echo "## Python Packages in torch"
  echo '```text'
  conda run -n torch python -c "import importlib
for name in ['torch', 'numpy', 'pandas', 'matplotlib', 'tqdm', 'scipy']:
    try:
        mod = importlib.import_module(name)
        print(f'{name}: ok ({getattr(mod, \"__version__\", \"unknown\")})')
    except Exception as exc:
        print(f'{name}: missing or failed ({exc})')"
  echo '```'
  echo
  echo "## Vitis 2025.2 Tools"
  echo '```text'
  if [ -f /opt/Xilinx/Vitis/2025.2/settings64.sh ]; then
    # shellcheck disable=SC1091
    source /opt/Xilinx/Vitis/2025.2/settings64.sh
    echo "sourced /opt/Xilinx/Vitis/2025.2/settings64.sh"
  elif [ -f /opt/Xilinx/2025.2/Vitis/settings64.sh ]; then
    echo "/opt/Xilinx/Vitis/2025.2/settings64.sh not found"
    # shellcheck disable=SC1091
    source /opt/Xilinx/2025.2/Vitis/settings64.sh
    echo "sourced /opt/Xilinx/2025.2/Vitis/settings64.sh"
  else
    echo "/opt/Xilinx/Vitis/2025.2/settings64.sh not found"
    echo "/opt/Xilinx/2025.2/Vitis/settings64.sh not found"
  fi
  which vitis || echo "vitis not found"
  which v++ || echo "v++ not found"
  which vitis_hls || echo "vitis_hls not found"
  echo '```'
} > "${OUT_FILE}" 2>&1

echo "Environment check written to ${OUT_FILE}"
