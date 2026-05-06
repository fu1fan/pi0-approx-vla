#include "kernel.hpp"

static const gelu_t kGeluPwlY[17] = {
    -0.000070246f, -0.000616198f, -0.003637392f, -0.015084266f,
    -0.045402306f, -0.100428423f, -0.158808009f, -0.154285990f,
    0.000000000f, 0.345714010f, 0.841191991f, 1.399571577f,
    1.954597694f, 2.484915734f, 2.996362608f, 3.499383802f,
    3.999929754f};

static gelu_t gelu_pwl_scalar(gelu_t x) {
#pragma HLS INLINE
    if (x <= static_cast<gelu_t>(-4.0)) {
        return static_cast<gelu_t>(0.0);
    }
    if (x >= static_cast<gelu_t>(4.0)) {
        return x;
    }

    const gelu_acc_t scaled =
        (static_cast<gelu_acc_t>(x) + static_cast<gelu_acc_t>(4.0)) *
        static_cast<gelu_acc_t>(2.0);
    int idx = static_cast<int>(scaled);
    if (idx < 0) {
        idx = 0;
    }
    if (idx > 15) {
        idx = 15;
    }
    const gelu_acc_t frac = scaled - static_cast<gelu_acc_t>(idx);
    const gelu_acc_t y0 = static_cast<gelu_acc_t>(kGeluPwlY[idx]);
    const gelu_acc_t y1 = static_cast<gelu_acc_t>(kGeluPwlY[idx + 1]);
    return static_cast<gelu_t>(y0 + (y1 - y0) * frac);
}

extern "C" void gelu_pwl_kernel(
    const gelu_t input[GELU_LEN],
    gelu_t output[GELU_LEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=GELU_LEN
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=GELU_LEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

vector_loop:
    for (int i = 0; i < GELU_LEN; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = gelu_pwl_scalar(input[i]);
    }
}
