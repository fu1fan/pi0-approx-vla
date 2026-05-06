#include "kernel.hpp"

static const softmax_prob_t kExpLut[SOFTMAX_LUT_SIZE] = {
    0.000335463f, 0.000380884f, 0.000432455f, 0.000491009f,
    0.000557491f, 0.000632975f, 0.000718679f, 0.000815988f,
    0.000926472f, 0.001051915f, 0.001194343f, 0.001356056f,
    0.001539665f, 0.001748134f, 0.001984830f, 0.002253574f,
    0.002558705f, 0.002905151f, 0.003298506f, 0.003745120f,
    0.004252206f, 0.004827950f, 0.005481650f, 0.006223859f,
    0.007066564f, 0.008023369f, 0.009109726f, 0.010343173f,
    0.011743628f, 0.013333704f, 0.015139074f, 0.017188889f,
    0.019516248f, 0.022158728f, 0.025158998f, 0.028565501f,
    0.032433241f, 0.036824669f, 0.041810692f, 0.047471818f,
    0.053899455f, 0.061197387f, 0.069483451f, 0.078891441f,
    0.089573263f, 0.101701392f, 0.115471659f, 0.131106405f,
    0.148858081f, 0.169013315f, 0.191897549f, 0.217880284f,
    0.247381055f, 0.280876202f, 0.318906557f, 0.362086185f,
    0.411112291f, 0.466776482f, 0.529977548f, 0.601735976f,
    0.683210423f, 0.775716428f, 0.880747653f, 1.000000000f};

static softmax_prob_t lut_exp_neg8_0(softmax_acc_t x) {
#pragma HLS INLINE
    if (x < static_cast<softmax_acc_t>(-8.0)) {
        x = static_cast<softmax_acc_t>(-8.0);
    }
    if (x > static_cast<softmax_acc_t>(0.0)) {
        x = static_cast<softmax_acc_t>(0.0);
    }
    const softmax_acc_t scaled =
        (x + static_cast<softmax_acc_t>(8.0)) *
        static_cast<softmax_acc_t>((SOFTMAX_LUT_SIZE - 1) / 8.0);
    int idx = static_cast<int>(scaled);
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= SOFTMAX_LUT_SIZE) {
        idx = SOFTMAX_LUT_SIZE - 1;
    }
    return kExpLut[idx];
}

extern "C" void lut_softmax_kernel(
    const softmax_in_t input[SOFTMAX_ROWS][SOFTMAX_LEN],
    softmax_prob_t output[SOFTMAX_ROWS][SOFTMAX_LEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=SOFTMAX_ROWS * SOFTMAX_LEN
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=SOFTMAX_ROWS * SOFTMAX_LEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    softmax_prob_t exp_buf[SOFTMAX_LEN];
#pragma HLS BIND_STORAGE variable=exp_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=exp_buf cyclic factor=8 dim=1

row_loop:
    for (int r = 0; r < SOFTMAX_ROWS; ++r) {
        softmax_in_t row_max = input[r][0];
    max_loop:
        for (int c = 1; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            const softmax_in_t value = input[r][c];
            if (value > row_max) {
                row_max = value;
            }
        }

        softmax_acc_t sum = 0;
    exp_loop:
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            const softmax_acc_t shifted =
                static_cast<softmax_acc_t>(input[r][c]) - static_cast<softmax_acc_t>(row_max);
            const softmax_prob_t e = lut_exp_neg8_0(shifted);
            exp_buf[c] = e;
            sum += static_cast<softmax_acc_t>(e);
        }

        const softmax_acc_t inv_sum = static_cast<softmax_acc_t>(1.0) / sum;
    norm_loop:
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
#pragma HLS PIPELINE II=1
            output[r][c] = static_cast<softmax_prob_t>(
                static_cast<softmax_acc_t>(exp_buf[c]) * inv_sum);
        }
    }
}
