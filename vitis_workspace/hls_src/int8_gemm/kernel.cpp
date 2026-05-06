#include "kernel.hpp"

static gemm_o16_t saturate_i16(int64_t value) {
#pragma HLS INLINE
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return static_cast<gemm_o16_t>(value);
}

static gemm_o16_t requantize_i16(gemm_acc_t acc, int32_t scale_q15, int32_t shift) {
#pragma HLS INLINE
    int64_t scaled = static_cast<int64_t>(acc) * static_cast<int64_t>(scale_q15);
    if (shift > 0) {
        const int64_t rounding = static_cast<int64_t>(1) << (shift - 1);
        if (scaled >= 0) {
            scaled = (scaled + rounding) >> shift;
        } else {
            scaled = -(((-scaled) + rounding) >> shift);
        }
    }
    return saturate_i16(scaled);
}

extern "C" void int8_gemm_kernel(
    const gemm_i8_t input[GEMM_M][GEMM_K],
    const gemm_i8_t weight[GEMM_K][GEMM_N],
    const gemm_acc_t bias[GEMM_N],
    gemm_o16_t output[GEMM_M][GEMM_N],
    int32_t scale_q15,
    int32_t shift) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=GEMM_M * GEMM_K
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=GEMM_K * GEMM_N
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem2 depth=GEMM_N
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=GEMM_M * GEMM_N
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=scale_q15 bundle=control
#pragma HLS INTERFACE s_axilite port=shift bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    gemm_acc_t acc_tile[GEMM_N];
#pragma HLS BIND_STORAGE variable=acc_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=acc_tile cyclic factor=16 dim=1

row_loop:
    for (int m = 0; m < GEMM_M; ++m) {
    init_loop:
        for (int n = 0; n < GEMM_N; ++n) {
#pragma HLS PIPELINE II=1
            acc_tile[n] = bias[n];
        }

    k_loop:
        for (int k = 0; k < GEMM_K; ++k) {
            const gemm_i8_t a_value = input[m][k];
        n_loop:
            for (int n = 0; n < GEMM_N; ++n) {
#pragma HLS PIPELINE II=1
                acc_tile[n] += static_cast<gemm_acc_t>(a_value) *
                               static_cast<gemm_acc_t>(weight[k][n]);
            }
        }

    store_loop:
        for (int n = 0; n < GEMM_N; ++n) {
#pragma HLS PIPELINE II=1
            output[m][n] = requantize_i16(acc_tile[n], scale_q15, shift);
        }
    }
}
