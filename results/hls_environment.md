# Vitis HLS Environment Audit

Date: 2026-05-07

## Repository State

- Repository: `fu1fan/pi0-approx-vla`
- Working directory: `/home/fu1fan/Develop/PROJECTS/pi0-approx-vla`
- Existing HLS source directory status: `vitis_workspace/hls_src/` is present and contains module-level kernels plus exact before/after baselines.
- Vitis Unified components detected under `vitis_workspace/`: `int8_gemm`, `exact_softmax`, `lut_softmax`, `exact_gelu`, `gelu_pwl`, `exact_rmsnorm`, `rmsnorm_rsqrt`, and `fixed_projector_tile`.
- The exact baseline comparison pass added only source/config/report artifacts needed for module-level benchmarking; Vitis build directories and generated kernel work directories remain ignored.

## Tool Availability

| Tool | Result |
| --- | --- |
| `vitis --version` | Vitis v2025.2, SW Build 6295257; direct IDE command still reports HOME configuration space pressure without `XILINX_VITIS_DATA_DIR` |
| `v++ --version` | v++ v2025.2, SW Build 6295257 |
| `vitis_hls -version` | Not found in `PATH` |
| `v++ --mode hls --help` | Available |

## Notes

- Direct `vitis --version` prints the correct Vitis Unified version but also reports: `The minimum disk space of 100MB not available in User HOME directory`.
- Filesystem-level `df -h` shows sufficient disk space on `/home`, `/tmp`, and the repository mount. The Vitis warning is therefore likely related to Vitis' chosen HOME/config data location or quota check rather than actual block-device free space.
- Setting `XILINX_VITIS_DATA_DIR=/tmp/vitis_data` is used by `scripts/run_all_hls.py` for Vitis Unified/v++ runs.
- Because `vitis_hls` is unavailable, batch synthesis should use Vitis Unified / `v++ --mode hls` where possible. Classic `run_hls.tcl` files are still useful as source-list and reproducibility templates.

## Recommended Command Prefix

```bash
env XILINX_VITIS_DATA_DIR=/tmp/vitis_data v++ --mode hls --config <component>/hls_config.cfg
```

If synthesis fails for a kernel, the failure reason should be recorded in `results/hls_kernel_summary.md` and `results/csv/hls_kernel_summary.csv` instead of being replaced by estimated results.
