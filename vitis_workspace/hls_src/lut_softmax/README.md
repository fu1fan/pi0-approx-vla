# lut_softmax

Vector softmax kernel using max subtraction and a small LUT approximation for `exp`.

- Vector length: `VEC_LEN=128`
- Input and output: `ap_fixed<16,6>`
- Top function: `lut_softmax`
- The normalization uses `/ sum` directly for clarity. A reciprocal LUT or Newton-Raphson reciprocal is a natural follow-up optimization if division is too costly in synthesis.

## Vitis Unified 2025.2

Create an HLS component, add `lut_softmax.cpp`, `lut_softmax.h`, and `testbench.cpp`, set top function to `lut_softmax`, run C simulation, then synthesis.

The `run_hls.tcl` file is a reference for users who have a compatible `vitis_hls` command.
