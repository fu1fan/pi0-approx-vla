#ifndef PI0_APPROX_VLA_FIXED_PROJECTOR_TILE_KERNEL_HPP
#define PI0_APPROX_VLA_FIXED_PROJECTOR_TILE_KERNEL_HPP

#ifndef PROJ_TOKENS
#define PROJ_TOKENS 64
#endif

#ifndef PROJ_IN_DIM
#define PROJ_IN_DIM 1152
#endif

#ifndef PROJ_OUT_DIM
#define PROJ_OUT_DIM 256
#endif

#ifdef HLS_NO_AP_FIXED
typedef float proj_t;
typedef float proj_acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> proj_t;
typedef ap_fixed<40, 16> proj_acc_t;
#endif

extern "C" void fixed_projector_tile_kernel(
    const proj_t input[PROJ_TOKENS][PROJ_IN_DIM],
    const proj_t weight[PROJ_IN_DIM][PROJ_OUT_DIM],
    const proj_t bias[PROJ_OUT_DIM],
    proj_t output[PROJ_TOKENS][PROJ_OUT_DIM]);

#endif
