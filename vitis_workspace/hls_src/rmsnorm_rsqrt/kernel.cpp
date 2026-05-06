#include "kernel.hpp"

static const rms_acc_t kRsqrtLut[RSQRT_LUT_SIZE] = {
    2.000000000f, 1.641844138f, 1.425758354f, 1.277332747f,
    1.167320591f, 1.081578162f, 1.012320793f, 0.954863711f,
    0.906196476f, 0.864284647f, 0.827697332f, 0.795394909f,
    0.766601412f, 0.740724352f, 0.717302462f, 0.695970545f,
    0.676435197f, 0.658457617f, 0.641841205f, 0.626422432f,
    0.612064013f, 0.598649733f, 0.586080459f, 0.574271046f,
    0.563147893f, 0.552647011f, 0.542712463f, 0.533295106f,
    0.524351569f, 0.515843407f, 0.507736406f, 0.500000000f};

static rms_acc_t rsqrt_lut_init(rms_acc_t x) {
#pragma HLS INLINE
    if (x < static_cast<rms_acc_t>(0.25)) {
        x = static_cast<rms_acc_t>(0.25);
    }
    if (x > static_cast<rms_acc_t>(4.0)) {
        x = static_cast<rms_acc_t>(4.0);
    }
    const rms_acc_t scaled =
        (x - static_cast<rms_acc_t>(0.25)) *
        static_cast<rms_acc_t>((RSQRT_LUT_SIZE - 1) / 3.75);
    int idx = static_cast<int>(scaled);
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= RSQRT_LUT_SIZE) {
        idx = RSQRT_LUT_SIZE - 1;
    }
    return kRsqrtLut[idx];
}

static rms_acc_t nr_rsqrt_step(rms_acc_t x, rms_acc_t y) {
#pragma HLS INLINE
    const rms_acc_t half = static_cast<rms_acc_t>(0.5);
    const rms_acc_t three_half = static_cast<rms_acc_t>(1.5);
    return y * (three_half - half * x * y * y);
}

extern "C" void rmsnorm_rsqrt_kernel(
    const rms_t input[RMSNORM_HIDDEN],
    const rms_t weight[RMSNORM_HIDDEN],
    rms_t output_nr1[RMSNORM_HIDDEN],
    rms_t output_nr2[RMSNORM_HIDDEN]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE m_axi port=output_nr1 offset=slave bundle=gmem2 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE m_axi port=output_nr2 offset=slave bundle=gmem3 depth=RMSNORM_HIDDEN
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=output_nr1 bundle=control
#pragma HLS INTERFACE s_axilite port=output_nr2 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    rms_acc_t sum_sq = 0;
sum_loop:
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
#pragma HLS PIPELINE II=1
        const rms_acc_t value = static_cast<rms_acc_t>(input[i]);
        sum_sq += value * value;
    }

    const rms_acc_t mean_sq =
        sum_sq / static_cast<rms_acc_t>(RMSNORM_HIDDEN) + static_cast<rms_acc_t>(1.0e-5);
    const rms_acc_t y0 = rsqrt_lut_init(mean_sq);
    const rms_acc_t y1 = nr_rsqrt_step(mean_sq, y0);
    const rms_acc_t y2 = nr_rsqrt_step(mean_sq, y1);

norm_loop:
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
#pragma HLS PIPELINE II=1
        const rms_acc_t scaled = static_cast<rms_acc_t>(input[i]) * static_cast<rms_acc_t>(weight[i]);
        output_nr1[i] = static_cast<rms_t>(scaled * y1);
        output_nr2[i] = static_cast<rms_t>(scaled * y2);
    }
}
