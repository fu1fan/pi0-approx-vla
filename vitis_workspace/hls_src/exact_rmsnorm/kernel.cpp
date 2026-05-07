#include "kernel.hpp"

#ifdef HLS_NO_AP_FIXED
#include <cmath>
static rms_exact_t exact_sqrt(rms_exact_t x) {
#pragma HLS INLINE
    return std::sqrt(x);
}
#else
#include <hls_math.h>
static rms_exact_t exact_sqrt(rms_exact_t x) {
#pragma HLS INLINE
    return hls::sqrt(x);
}
#endif

extern "C" void exact_rmsnorm_kernel(
    const rms_exact_t input[RMSNORM_HIDDEN],
    const rms_exact_t weight[RMSNORM_HIDDEN],
    rms_exact_t output[RMSNORM_HIDDEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    rms_exact_t sum_sq = 0.0f;
sum_loop:
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
#pragma HLS PIPELINE II=1
        const rms_exact_t value = input[i];
        sum_sq += value * value;
    }

    const rms_exact_t mean_sq = sum_sq / static_cast<rms_exact_t>(RMSNORM_HIDDEN) + 1.0e-5f;
    const rms_exact_t inv_rms = 1.0f / exact_sqrt(mean_sq);

norm_loop:
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = input[i] * weight[i] * inv_rms;
    }
}
