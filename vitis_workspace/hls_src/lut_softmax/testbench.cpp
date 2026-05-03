#include <cmath>
#include <iostream>

#include "lut_softmax.h"

int main() {
    data_t x[VEC_LEN];
    data_t y[VEC_LEN];
    double golden[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        x[i] = ((i * 7) % 37) / 8.0 - 2.0;
    }

    lut_softmax(x, y);

    double max_val = (double)x[0];
    for (int i = 1; i < VEC_LEN; i++) {
        max_val = std::max(max_val, (double)x[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < VEC_LEN; i++) {
        golden[i] = std::exp((double)x[i] - max_val);
        sum += golden[i];
    }

    double max_error = 0.0;
    double mean_error = 0.0;
    for (int i = 0; i < VEC_LEN; i++) {
        golden[i] /= sum;
        double err = std::fabs((double)y[i] - golden[i]);
        max_error = std::max(max_error, err);
        mean_error += err;
    }
    mean_error /= VEC_LEN;

    std::cout << "max_error=" << max_error << std::endl;
    std::cout << "mean_error=" << mean_error << std::endl;
    std::cout << (max_error < 0.02 ? "PASS" : "FAIL") << std::endl;
    return max_error < 0.02 ? 0 : 1;
}
