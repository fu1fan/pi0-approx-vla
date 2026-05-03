#ifndef LUT_SOFTMAX_H
#define LUT_SOFTMAX_H

#include <ap_fixed.h>

#define VEC_LEN 128
#define LUT_SIZE 16

typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<24, 8> acc_t;

void lut_softmax(const data_t x[VEC_LEN], data_t y[VEC_LEN]);

#endif
