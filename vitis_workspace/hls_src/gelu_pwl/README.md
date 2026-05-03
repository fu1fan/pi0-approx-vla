# gelu_pwl

Piecewise-linear GELU approximation for a fixed vector.

- Vector length: `VEC_LEN=128`
- Input and output: `ap_fixed<16,6>`
- Top function: `gelu_pwl`
- Approximation behavior:
  - `x < -3 -> 0`
  - `x > 3 -> x`
  - Middle range uses linear segments

## Vitis Unified 2025.2

Create an HLS component, add the source and testbench files in this directory, set top function to `gelu_pwl`, then run C simulation and synthesis.

The TCL file is retained as a reference for compatible command-line HLS flows.
