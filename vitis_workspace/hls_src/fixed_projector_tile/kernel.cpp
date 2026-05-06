#include "kernel.hpp"

extern "C" void fixed_projector_tile_kernel(
    const proj_t input[PROJ_TOKENS][PROJ_IN_DIM],
    const proj_t weight[PROJ_IN_DIM][PROJ_OUT_DIM],
    const proj_t bias[PROJ_OUT_DIM],
    proj_t output[PROJ_TOKENS][PROJ_OUT_DIM]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=PROJ_TOKENS * PROJ_IN_DIM
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=PROJ_IN_DIM * PROJ_OUT_DIM
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem2 depth=PROJ_OUT_DIM
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=PROJ_TOKENS * PROJ_OUT_DIM
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    proj_acc_t acc_tile[PROJ_OUT_DIM];
#pragma HLS BIND_STORAGE variable=acc_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=acc_tile cyclic factor=8 dim=1

token_loop:
    for (int t = 0; t < PROJ_TOKENS; ++t) {
    init_loop:
        for (int o = 0; o < PROJ_OUT_DIM; ++o) {
#pragma HLS PIPELINE II=1
            acc_tile[o] = static_cast<proj_acc_t>(bias[o]);
        }

    in_loop:
        for (int i = 0; i < PROJ_IN_DIM; ++i) {
            const proj_acc_t x = static_cast<proj_acc_t>(input[t][i]);
        out_loop:
            for (int o = 0; o < PROJ_OUT_DIM; ++o) {
#pragma HLS PIPELINE II=1
                acc_tile[o] += x * static_cast<proj_acc_t>(weight[i][o]);
            }
        }

    store_loop:
        for (int o = 0; o < PROJ_OUT_DIM; ++o) {
#pragma HLS PIPELINE II=1
            output[t][o] = static_cast<proj_t>(acc_tile[o]);
        }
    }
}
