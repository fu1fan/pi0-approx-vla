#ifndef INT8_LINEAR_H
#define INT8_LINEAR_H

#include <ap_int.h>

#define IN_DIM 128
#define OUT_DIM 64

typedef ap_int<8> act_t;
typedef ap_int<8> weight_t;
typedef ap_int<32> bias_t;
typedef ap_int<16> out_t;

void int8_linear(const act_t x[IN_DIM],
                 const weight_t w[OUT_DIM][IN_DIM],
                 const bias_t b[OUT_DIM],
                 out_t y[OUT_DIM]);

#endif
