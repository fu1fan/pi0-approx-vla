#ifndef PI0_APPROX_VLA_INT8_GEMM_KERNEL_HPP
#define PI0_APPROX_VLA_INT8_GEMM_KERNEL_HPP

#include <stdint.h>

#ifndef GEMM_M
#define GEMM_M 50
#endif

#ifndef GEMM_K
#define GEMM_K 32
#endif

#ifndef GEMM_N
#define GEMM_N 1024
#endif

typedef int8_t gemm_i8_t;
typedef int16_t gemm_o16_t;
typedef int32_t gemm_acc_t;

extern "C" void int8_gemm_kernel(
    const gemm_i8_t input[GEMM_M][GEMM_K],
    const gemm_i8_t weight[GEMM_K][GEMM_N],
    const gemm_acc_t bias[GEMM_N],
    gemm_o16_t output[GEMM_M][GEMM_N],
    int32_t scale_q15,
    int32_t shift);

#endif
