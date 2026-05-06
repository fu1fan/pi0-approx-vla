# RMSNorm + Approximate Rsqrt HLS Benchmark

This kernel isolates Transformer RMSNorm for pi0-style VLM and action-expert blocks. It does not deploy pi0 end to end.

## Default Vector

- Hidden size: `1024`
- Supported by macro rebuild: `RMSNORM_HIDDEN=2048`
- Input/output type: `ap_fixed<16,6>`
- Accumulator: `ap_fixed<40,16>`
- Rsqrt approximation: 32-entry LUT over mean-square range `[0.25, 4.0]` followed by Newton-Raphson 1-step and 2-step branches

## Top Function

`rmsnorm_rsqrt_kernel`

The kernel writes two output vectors:

- `output_nr1`: LUT initial value + one Newton-Raphson step
- `output_nr2`: LUT initial value + two Newton-Raphson steps

## Local C++ Smoke Test

```bash
g++ -std=c++17 -DHLS_NO_AP_FIXED kernel.cpp tb.cpp -o /tmp/rmsnorm_rsqrt_tb
/tmp/rmsnorm_rsqrt_tb
```

## Vitis Unified Batch Flow

```bash
cd vitis_workspace/rmsnorm_rsqrt
env XILINX_VITIS_DATA_DIR=/tmp/vitis_data v++ --mode hls --config hls_config.cfg
```
