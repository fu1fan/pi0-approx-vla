#include "fixed_projector.h"

void fixed_projector(const data_t x[IN_DIM],
                     const data_t w[OUT_DIM][IN_DIM],
                     const data_t b[OUT_DIM],
                     data_t y[OUT_DIM]) {
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
        acc_t acc = b[o];
    IN_LOOP:
        for (int i = 0; i < IN_DIM; i++) {
#pragma HLS UNROLL factor=4
            acc += (acc_t)x[i] * (acc_t)w[o][i];
        }
        y[o] = (data_t)acc;
    }
}
