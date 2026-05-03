#include <cmath>
#include <iostream>

#include "fixed_projector.h"

int main() {
    data_t x[IN_DIM];
    data_t w[OUT_DIM][IN_DIM];
    data_t b[OUT_DIM];
    data_t y[OUT_DIM];

    for (int i = 0; i < IN_DIM; i++) {
        x[i] = ((i % 23) - 11) / 16.0;
    }
    for (int o = 0; o < OUT_DIM; o++) {
        b[o] = ((o % 7) - 3) / 32.0;
        for (int i = 0; i < IN_DIM; i++) {
            w[o][i] = (((o + i * 3) % 19) - 9) / 64.0;
        }
    }

    fixed_projector(x, w, b, y);

    double max_error = 0.0;
    double mean_error = 0.0;
    for (int o = 0; o < OUT_DIM; o++) {
        double golden = (double)b[o];
        for (int i = 0; i < IN_DIM; i++) {
            golden += (double)x[i] * (double)w[o][i];
        }
        double err = std::fabs((double)y[o] - golden);
        max_error = std::max(max_error, err);
        mean_error += err;
    }
    mean_error /= OUT_DIM;

    std::cout << "max_error=" << max_error << std::endl;
    std::cout << "mean_error=" << mean_error << std::endl;
    std::cout << (max_error < 0.02 ? "PASS" : "FAIL") << std::endl;
    return max_error < 0.02 ? 0 : 1;
}
