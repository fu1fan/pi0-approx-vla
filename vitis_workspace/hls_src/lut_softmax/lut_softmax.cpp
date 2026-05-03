#include "lut_softmax.h"

static data_t exp_lut(data_t x) {
    const data_t lut[LUT_SIZE] = {
        0.00033546, 0.00057258, 0.00097774, 0.00166901,
        0.00284930, 0.00486427, 0.00830479, 0.01417930,
        0.02420700, 0.04131000, 0.07052370, 0.12040900,
        0.20555700, 0.35091900, 0.59929600, 1.00000000
    };
    data_t xc = x;
    if (xc < -8) xc = -8;
    if (xc > 0) xc = 0;
    data_t pos = (xc + 8) * (LUT_SIZE - 1) / 8;
    int idx = (int)(pos + (data_t)0.5);
    if (idx < 0) idx = 0;
    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return lut[idx];
}

void lut_softmax(const data_t x[VEC_LEN], data_t y[VEC_LEN]) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t exps[VEC_LEN];
#pragma HLS ARRAY_PARTITION variable=exps cyclic factor=4

    data_t max_val = x[0];
MAX_LOOP:
    for (int i = 1; i < VEC_LEN; i++) {
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }

    acc_t sum = 0;
EXP_LOOP:
    for (int i = 0; i < VEC_LEN; i++) {
#pragma HLS PIPELINE II=1
        data_t e = exp_lut(x[i] - max_val);
        exps[i] = e;
        sum += e;
    }

NORM_LOOP:
    for (int i = 0; i < VEC_LEN; i++) {
#pragma HLS PIPELINE II=1
        y[i] = (data_t)(exps[i] / sum);
    }
}
