#ifndef GELU_PWL_H
#define GELU_PWL_H

#include <ap_fixed.h>

#define VEC_LEN 128

typedef ap_fixed<16, 6> data_t;

void gelu_pwl(const data_t x[VEC_LEN], data_t y[VEC_LEN]);

#endif
