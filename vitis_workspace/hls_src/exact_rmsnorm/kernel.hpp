#ifndef PI0_APPROX_VLA_EXACT_RMSNORM_KERNEL_HPP
#define PI0_APPROX_VLA_EXACT_RMSNORM_KERNEL_HPP

#ifndef RMSNORM_HIDDEN
#define RMSNORM_HIDDEN 1024
#endif

typedef float rms_exact_t;

extern "C" void exact_rmsnorm_kernel(
    const rms_exact_t input[RMSNORM_HIDDEN],
    const rms_exact_t weight[RMSNORM_HIDDEN],
    rms_exact_t output[RMSNORM_HIDDEN]);

#endif
