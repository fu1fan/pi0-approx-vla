#include "gelu_pwl.h"

static data_t gelu_scalar(data_t x) {
    if (x < -3) return 0;
    if (x > 3) return x;
    if (x < -2) return (data_t)0.045 * (x + 3);
    if (x < -1) return (data_t)-0.045 + (data_t)-0.113 * (x + 2);
    if (x < 0) return (data_t)-0.1587 + (data_t)0.1587 * (x + 1);
    if (x < 1) return (data_t)0.5 * x;
    if (x < 2) return (data_t)0.8413 + (data_t)1.113 * (x - 1);
    return (data_t)1.9545 + (data_t)1.0455 * (x - 2);
}

void gelu_pwl(const data_t x[VEC_LEN], data_t y[VEC_LEN]) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

LOOP:
    for (int i = 0; i < VEC_LEN; i++) {
#pragma HLS PIPELINE II=1
        y[i] = gelu_scalar(x[i]);
    }
}
