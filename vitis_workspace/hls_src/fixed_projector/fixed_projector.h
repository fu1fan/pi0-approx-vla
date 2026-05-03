#ifndef FIXED_PROJECTOR_H
#define FIXED_PROJECTOR_H

#include <ap_fixed.h>

#define IN_DIM 128
#define OUT_DIM 64

typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<32, 12> acc_t;

void fixed_projector(const data_t x[IN_DIM],
                     const data_t w[OUT_DIM][IN_DIM],
                     const data_t b[OUT_DIM],
                     data_t y[OUT_DIM]);

#endif
