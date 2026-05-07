# HLS Optimization Before/After Comparison

This report compares optimization-before exact-style HLS baselines against the existing approximate module-level kernels. It does not include power and does not represent full pi0 deployment.

## Comparison Contract

- Softmax: float exp row-wise softmax baseline vs LUT exp softmax.
- GELU: float tanh GELU baseline vs 16-segment PWL GELU.
- RMSNorm: float sqrt RMSNorm baseline vs LUT-initialized Newton-Raphson rsqrt branches.
- Metrics come from HLS C-sim and C-synthesis reports when synthesis succeeds.

## Results

| comparison_group | baseline_kernel | optimized_kernel | shape | baseline_latency_cycles | optimized_latency_cycles | latency_speedup | baseline_clock_ns | optimized_clock_ns | baseline_estimated_time_ns | optimized_estimated_time_ns | time_speedup | baseline_LUT | optimized_LUT | LUT_delta_pct | baseline_DSP | optimized_DSP | DSP_delta_pct | MSE | cosine | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| softmax | exact_softmax | lut_softmax | rows4_len128 | 1727 | 1867 | 0.925013 | 15.145 | 7.300 | 2.615542e+04 | 1.362910e+04 | 1.919086 | 4133 | 3692 | -10.670215 | 14 | 2 | -85.714286 | 3.723352544601e-07 | 0.999311199515 | passed |
| gelu | exact_gelu | gelu_pwl | len4096 | 4159 | 4114 | 1.010938 | 7.300 | 7.300 | 3.036070e+04 | 3.003220e+04 | 1.010938 | 7854 | 2301 | -70.702827 | 60 | 2 | -96.666667 | 3.836425107314e-05 | 0.999995307692 | passed |
| rmsnorm_nr1 | exact_rmsnorm | rmsnorm_rsqrt | hidden1024 | 2094 | 2080 | 1.006731 | 15.145 | 7.300 | 3.171363e+04 | 1.518400e+04 | 2.088622 | 3657 | 5649 | 54.470878 | 10 | 64 | 540.000000 | 2.950255249681e-07 | 0.999999855028 | passed |
| rmsnorm_nr2 | exact_rmsnorm | rmsnorm_rsqrt | hidden1024 | 2094 | 2080 | 1.006731 | 15.145 | 7.300 | 3.171363e+04 | 1.518400e+04 | 2.088622 | 3657 | 5649 | 54.470878 | 10 | 64 | 540.000000 | 3.014300453104e-07 | 0.999999848934 | passed |

## Notes

- All comparison pairs synthesized and parsed successfully.

## Interpretation

- `latency_speedup` is `baseline_latency / optimized_latency`; values above 1 mean the optimized kernel has fewer cycles.
- `time_speedup` uses `latency_cycles * estimated_clock_ns`; this captures cases where an approximate kernel has similar cycles but a better estimated clock.
- Resource delta columns are `(optimized - baseline) / baseline * 100`; negative values mean the optimized kernel used fewer resources.
- Error columns report the optimized kernel error versus the Python/C++ golden reference. Baseline numeric error is retained in `baseline_MSE` and `baseline_cosine` in the CSV.
