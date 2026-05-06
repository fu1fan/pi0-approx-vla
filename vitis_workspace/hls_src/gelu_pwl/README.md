# PWL GELU HLS Benchmark

This kernel isolates FFN activation approximation for pi0-style VLM and action-expert MLP blocks. It does not replace the full FFN or deploy pi0 end to end.

## Default Vector

- Length: `4096`, matching action-expert FFN hidden scale
- Supported by macro rebuild: `GELU_LEN=16384` for VLM FFN hidden streaming-vector tests
- Input/output type: `ap_fixed<16,6>`
- Approximation: 16 uniform PWL segments over `[-4, 4]`
- Outside range: output `0` for `x <= -4`, identity for `x >= 4`

## Top Function

`gelu_pwl_kernel`

## Local C++ Smoke Test

```bash
g++ -std=c++17 -DHLS_NO_AP_FIXED kernel.cpp tb.cpp -o /tmp/gelu_pwl_tb
/tmp/gelu_pwl_tb
```

## Vitis Unified Batch Flow

```bash
cd vitis_workspace/gelu_pwl
env XILINX_VITIS_DATA_DIR=/tmp/vitis_data v++ --mode hls --config hls_config.cfg
```
