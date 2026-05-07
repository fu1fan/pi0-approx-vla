#include "kernel.hpp"

#ifdef HLS_NO_AP_FIXED
#include <cmath>
static softmax_exact_acc_t exact_exp(softmax_exact_acc_t x) {
#pragma HLS INLINE
    return std::exp(x);
}
#else
#include <hls_math.h>
static softmax_exact_acc_t exact_exp(softmax_exact_acc_t x) {
#pragma HLS INLINE
    return hls::exp(x);
}
#endif

extern "C" void exact_softmax_kernel(
    const softmax_exact_in_t input[SOFTMAX_ROWS][SOFTMAX_LEN],
    softmax_exact_prob_t output[SOFTMAX_ROWS][SOFTMAX_LEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=SOFTMAX_ROWS * SOFTMAX_LEN
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=SOFTMAX_ROWS * SOFTMAX_LEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    softmax_exact_acc_t exp_buf[SOFTMAX_LEN];
#pragma HLS BIND_STORAGE variable=exp_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=exp_buf cyclic factor=8 dim=1

row_loop:
    for (int r = 0; r < SOFTMAX_ROWS; ++r) {
        softmax_exact_acc_t row_max = input[r][0];
    max_loop:
        for (int c = 1; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            const softmax_exact_acc_t value = input[r][c];
            if (value > row_max) {
                row_max = value;
            }
        }

        softmax_exact_acc_t sum = 0.0f;
    exp_loop:
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            const softmax_exact_acc_t shifted = input[r][c] - row_max;
            const softmax_exact_acc_t e = exact_exp(shifted);
            exp_buf[c] = e;
            sum += e;
        }

        const softmax_exact_acc_t inv_sum = 1.0f / sum;
    norm_loop:
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            output[r][c] = exp_buf[c] * inv_sum;
        }
    }
}
