# Vitis HLS Environment Audit

Date: 2026-05-06

## Repository State

- Repository: `fu1fan/pi0-approx-vla`
- Working directory: `/home/fu1fan/Develop/PROJECTS/pi0-approx-vla`
- Existing HLS source directory status: `vitis_workspace/hls_src/` is absent in the current worktree.
- Existing Vitis Unified components detected under `vitis_workspace/`: `int8_gemm`, `lut_softmax`, `gelu_pwl`, `rmsnorm_rsqrt`.
- The worktree already contained modified Vitis IDE metadata and deleted older tracked HLS source files before this HLS pass. Those pre-existing changes are not treated as generated evidence and are not included in this stage commit.

## Tool Availability

| Tool | Result |
| --- | --- |
| `vitis --version` | Vitis v2025.2, SW Build 6295257 |
| `v++ --version` | v++ v2025.2, SW Build 6295257 |
| `vitis_hls -version` | Not found in `PATH` |
| `v++ --mode hls --help` | Available |

## Notes

- Direct `vitis --version` prints the correct Vitis Unified version but also reports: `The minimum disk space of 100MB not available in User HOME directory`.
- Filesystem-level `df -h` shows sufficient disk space on `/home`, `/tmp`, and the repository mount. The Vitis warning is therefore likely related to Vitis' chosen HOME/config data location or quota check rather than actual block-device free space.
- Setting `XILINX_VITIS_DATA_DIR=/tmp/vitis_data` allows `vitis --version` to run without the HOME-space warning.
- Because `vitis_hls` is unavailable, batch synthesis should use Vitis Unified / `v++ --mode hls` where possible. Classic `run_hls.tcl` files are still useful as source-list and reproducibility templates.

## Recommended Command Prefix

```bash
env XILINX_VITIS_DATA_DIR=/tmp/vitis_data v++ --mode hls --config <component>/hls_config.cfg
```

If synthesis fails for a kernel, the failure reason should be recorded in `results/hls_kernel_summary.md` and `results/csv/hls_kernel_summary.csv` instead of being replaced by estimated results.
