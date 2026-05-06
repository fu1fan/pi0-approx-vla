# LUT Softmax HLS Benchmark

This kernel isolates row-wise attention score normalization for pi0-style VLM and action-expert attention. It does not implement full attention or a full VLA model.

## Default Tile

- Rows: `4`
- Vector length: `128`
- Supported by macro rebuild: `SOFTMAX_LEN=50`, `128`, or `256`
- Input range after max subtraction: clamped to `[-8, 0]`
- Exp approximation: 64-entry LUT over `[-8, 0]`
- Output: fixed-point probability type

## Top Function

`lut_softmax_kernel`

## Local C++ Smoke Test

```bash
g++ -std=c++17 -DHLS_NO_AP_FIXED kernel.cpp tb.cpp -o /tmp/lut_softmax_tb
/tmp/lut_softmax_tb
```

## Vitis Unified Batch Flow

```bash
cd vitis_workspace/lut_softmax
env XILINX_VITIS_DATA_DIR=/tmp/vitis_data v++ --mode hls --config hls_config.cfg
```
