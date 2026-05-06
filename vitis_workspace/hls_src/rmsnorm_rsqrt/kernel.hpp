#ifndef PI0_APPROX_VLA_RMSNORM_RSQRT_KERNEL_HPP
#define PI0_APPROX_VLA_RMSNORM_RSQRT_KERNEL_HPP

#ifndef RMSNORM_HIDDEN
#define RMSNORM_HIDDEN 1024
#endif

#ifndef RSQRT_LUT_SIZE
#define RSQRT_LUT_SIZE 32
#endif

#ifdef HLS_NO_AP_FIXED
typedef float rms_t;
typedef float rms_acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> rms_t;
typedef ap_fixed<40, 16> rms_acc_t;
#endif

extern "C" void rmsnorm_rsqrt_kernel(
    const rms_t input[RMSNORM_HIDDEN],
    const rms_t weight[RMSNORM_HIDDEN],
    rms_t output_nr1[RMSNORM_HIDDEN],
    rms_t output_nr2[RMSNORM_HIDDEN]);

#endif
