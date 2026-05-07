#ifndef PI0_APPROX_VLA_EXACT_SOFTMAX_KERNEL_HPP
#define PI0_APPROX_VLA_EXACT_SOFTMAX_KERNEL_HPP

#ifndef SOFTMAX_ROWS
#define SOFTMAX_ROWS 4
#endif

#ifndef SOFTMAX_LEN
#define SOFTMAX_LEN 128
#endif

typedef float softmax_exact_in_t;
typedef float softmax_exact_prob_t;
typedef float softmax_exact_acc_t;

extern "C" void exact_softmax_kernel(
    const softmax_exact_in_t input[SOFTMAX_ROWS][SOFTMAX_LEN],
    softmax_exact_prob_t output[SOFTMAX_ROWS][SOFTMAX_LEN]);

#endif
