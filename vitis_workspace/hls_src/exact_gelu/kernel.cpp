#include "kernel.hpp"

#ifdef HLS_NO_AP_FIXED
#include <cmath>
static gelu_exact_t exact_tanh(gelu_exact_t x) {
#pragma HLS INLINE
    return std::tanh(x);
}
#else
#include <hls_math.h>
static gelu_exact_t exact_tanh(gelu_exact_t x) {
#pragma HLS INLINE
    return hls::tanh(x);
}
#endif

static gelu_exact_t exact_gelu_scalar(gelu_exact_t x) {
#pragma HLS INLINE
    const gelu_exact_t alpha = 0.7978845608028654f;
    const gelu_exact_t cubic_coeff = 0.044715f;
    const gelu_exact_t x3 = x * x * x;
    const gelu_exact_t inner = alpha * (x + cubic_coeff * x3);
    return 0.5f * x * (1.0f + exact_tanh(inner));
}

extern "C" void exact_gelu_kernel(
    const gelu_exact_t input[GELU_LEN],
    gelu_exact_t output[GELU_LEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=GELU_LEN
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=GELU_LEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

vector_loop:
    for (int i = 0; i < GELU_LEN; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = exact_gelu_scalar(input[i]);
    }
}
