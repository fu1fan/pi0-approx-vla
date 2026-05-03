# fixed_projector

Scaled-down fixed-point VLM projector kernel. It models a visual-token-to-language-embedding linear projection without reproducing the full `[256, 1152, 2048]` workload.

- Data type: `ap_fixed<16,6>`
- Accumulator: `ap_fixed<32,12>`
- Fixed size: `IN_DIM=128`, `OUT_DIM=64`
- Top function: `fixed_projector`

## Vitis Unified 2025.2

Create an HLS component, add the three source files in this directory, set the top function to `fixed_projector`, then run C simulation and synthesis.

The `run_hls.tcl` file is retained as a legacy command-flow reference.
