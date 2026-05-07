#ifndef PI0_APPROX_VLA_EXACT_GELU_KERNEL_HPP
#define PI0_APPROX_VLA_EXACT_GELU_KERNEL_HPP

#ifndef GELU_LEN
#define GELU_LEN 4096
#endif

typedef float gelu_exact_t;

extern "C" void exact_gelu_kernel(
    const gelu_exact_t input[GELU_LEN],
    gelu_exact_t output[GELU_LEN]);

#endif
