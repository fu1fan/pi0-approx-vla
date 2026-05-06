# Vitis HLS Kernel Benchmark Summary

This report is module-level only. It does not deploy full pi0 and does not run end-to-end VLA inference.

PyTorch/Python experiments provide module error trends with random or real pi0 weights. HLS kernels provide tile-level implementability evidence: C-sim numerical checks, synthesis status, and report-derived latency/resource fields when synthesis succeeds.

## Kernel Mapping

- `int8_gemm`: projector / QKV / FFN / state-action projection GEMM tile.
- `lut_softmax`: attention score normalization.
- `gelu_pwl`: FFN activation approximation.
- `rmsnorm_rsqrt`: Transformer RMSNorm reciprocal-square-root approximation.
- `fixed_projector_tile`: optional fixed-point visual projector tile.

Flow step reduction is an algorithm-level approximation and is intentionally not represented as an HLS kernel.

## Results

| kernel | variant | data_type | shape | latency_cycles | ii | estimated_clock_ns | LUT | FF | BRAM | DSP | mse | mae | kl | cosine | relative_l2 | hls_synth_status | parse_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_projector_tile | default | fixed16x6_acc40x16 | 64x1152x256 | 37783054 | 37783055 | 7.300 | 5477 | 3510 | 19 | 4 | 3.178747272625e-07 | 4.883789806627e-04 | NA | 0.999965279440 | 8.333756462524e-03 | passed | ok |
| gelu_pwl | default | fixed16x6 | len4096 | 4114 | 4096 | 7.300 | 2301 | 1806 | 1 | 2 | 3.836425107314e-05 | 3.405709433815e-03 | NA | 0.999995307692 | 3.070151235093e-03 | passed | ok |
| int8_gemm | default | int8_acc32_out16 | 50x32x1024 | 1742214 | 1742215 | 7.300 | 6953 | 4024 | 19 | 5 | 0.000000000000e+00 | 0.000000000000e+00 | NA | 1.000000000000 | 0.000000000000e+00 | passed | ok |
| lut_softmax | default | fixed16x6_prob18x2 | rows4_len128 | 1867 | 1868 | 7.300 | 3692 | 2727 | 9 | 2 | 3.723352544601e-07 | 2.567741628471e-04 | 1.746296664206e-03 | 0.999311199515 | 3.771399742744e-02 | passed | ok |
| rmsnorm_rsqrt | nr1 | fixed16x6_acc40x16 | hidden1024 | 2080 | 2081 | 7.300 | 5649 | 3857 | 2 | 64 | 2.950255249681e-07 | 4.613765880047e-04 | NA | 0.999999855028 | 5.446508053096e-04 | passed | ok |
| rmsnorm_rsqrt | nr2 | fixed16x6_acc40x16 | hidden1024 | 2080 | 2081 | 7.300 | 5649 | 3857 | 2 | 64 | 3.014300453104e-07 | 4.730451477809e-04 | NA | 0.999999848934 | 5.505308032742e-04 | passed | ok |

## Failures Or Downgrades

- No HLS synthesis failures recorded.

## Artifacts

- CSV: `results/csv/hls_kernel_summary.csv`
- Per-kernel status and report copies: `results/hls_reports/`
- HLS source: `vitis_workspace/hls_src/`
- Vitis Unified components: `vitis_workspace/<kernel>/`
