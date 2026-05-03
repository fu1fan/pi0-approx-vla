#include "int8_linear.h"

static out_t clamp_int16(ap_int<32> v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (out_t)v;
}

void int8_linear(const act_t x[IN_DIM],
                 const weight_t w[OUT_DIM][IN_DIM],
                 const bias_t b[OUT_DIM],
                 out_t y[OUT_DIM]) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

OUT_LOOP:
    for (int o = 0; o < OUT_DIM; o++) {
#pragma HLS PIPELINE II=1
        ap_int<32> acc = b[o];
    IN_LOOP:
        for (int i = 0; i < IN_DIM; i++) {
#pragma HLS UNROLL factor=4
            acc += (ap_int<16>)x[i] * (ap_int<16>)w[o][i];
        }
        y[o] = clamp_int16(acc);
    }
}
