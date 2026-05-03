#include <cmath>
#include <iostream>

#include "int8_linear.h"

static int clamp16(int v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return v;
}

int main() {
    act_t x[IN_DIM];
    weight_t w[OUT_DIM][IN_DIM];
    bias_t b[OUT_DIM];
    out_t y[OUT_DIM];

    for (int i = 0; i < IN_DIM; i++) {
        x[i] = (i % 17) - 8;
    }
    for (int o = 0; o < OUT_DIM; o++) {
        b[o] = (o % 13) - 6;
        for (int i = 0; i < IN_DIM; i++) {
            w[o][i] = ((o * 3 + i * 5) % 15) - 7;
        }
    }

    int8_linear(x, w, b, y);

    int max_error = 0;
    for (int o = 0; o < OUT_DIM; o++) {
        int golden = (int)b[o];
        for (int i = 0; i < IN_DIM; i++) {
            golden += (int)x[i] * (int)w[o][i];
        }
        golden = clamp16(golden);
        int err = std::abs((int)y[o] - golden);
        if (err > max_error) max_error = err;
    }

    std::cout << "max_error=" << max_error << std::endl;
    std::cout << (max_error == 0 ? "PASS" : "FAIL") << std::endl;
    return max_error == 0 ? 0 : 1;
}
