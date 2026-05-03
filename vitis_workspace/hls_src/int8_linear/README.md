# int8_linear

Module-level INT8 linear kernel for `Y = XW + b`.

- Input activation: `ap_int<8>`
- Weight: `ap_int<8>`
- Bias and accumulator: `ap_int<32>`
- Output: `ap_int<16>` with saturation
- Fixed size: `IN_DIM=128`, `OUT_DIM=64`
- Top function: `int8_linear`

## Vitis Unified 2025.2

Create an HLS component manually, add `int8_linear.cpp`, `int8_linear.h`, and `testbench.cpp`, then set the top function to `int8_linear`.

Run C simulation first. If it passes, run synthesis and copy the report summary to `results/hls_reports/`.

## TCL Reference

If a legacy `vitis_hls` command is available, this template may work:

```bash
source /opt/Xilinx/Vitis/2025.2/settings64.sh
vitis_hls -f run_hls.tcl
```

On Vitis Unified-only installs, treat `run_hls.tcl` as a source list and settings reference.
