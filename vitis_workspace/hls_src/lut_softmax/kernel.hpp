#ifndef PI0_APPROX_VLA_LUT_SOFTMAX_KERNEL_HPP
#define PI0_APPROX_VLA_LUT_SOFTMAX_KERNEL_HPP

#ifndef SOFTMAX_ROWS
#define SOFTMAX_ROWS 4
#endif

#ifndef SOFTMAX_LEN
#define SOFTMAX_LEN 128
#endif

#ifndef SOFTMAX_LUT_SIZE
#define SOFTMAX_LUT_SIZE 64
#endif

#ifdef HLS_NO_AP_FIXED
typedef float softmax_in_t;
typedef float softmax_prob_t;
typedef float softmax_acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> softmax_in_t;
typedef ap_fixed<18, 2> softmax_prob_t;
typedef ap_fixed<32, 10> softmax_acc_t;
#endif

extern "C" void lut_softmax_kernel(
    const softmax_in_t input[SOFTMAX_ROWS][SOFTMAX_LEN],
    softmax_prob_t output[SOFTMAX_ROWS][SOFTMAX_LEN]);

#endif
