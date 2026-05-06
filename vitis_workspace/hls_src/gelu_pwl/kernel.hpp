#ifndef PI0_APPROX_VLA_GELU_PWL_KERNEL_HPP
#define PI0_APPROX_VLA_GELU_PWL_KERNEL_HPP

#ifndef GELU_LEN
#define GELU_LEN 4096
#endif

#ifdef HLS_NO_AP_FIXED
typedef float gelu_t;
typedef float gelu_acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> gelu_t;
typedef ap_fixed<24, 8> gelu_acc_t;
#endif

extern "C" void gelu_pwl_kernel(
    const gelu_t input[GELU_LEN],
    gelu_t output[GELU_LEN]);

#endif
