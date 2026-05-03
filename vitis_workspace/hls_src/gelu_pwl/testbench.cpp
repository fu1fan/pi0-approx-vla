#include <cmath>
#include <iostream>

#include "gelu_pwl.h"

static double gelu_ref(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

int main() {
    data_t x[VEC_LEN];
    data_t y[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        x[i] = -5.0 + 10.0 * i / (VEC_LEN - 1);
    }

    gelu_pwl(x, y);

    double max_error = 0.0;
    double mean_error = 0.0;
    for (int i = 0; i < VEC_LEN; i++) {
        double err = std::fabs((double)y[i] - gelu_ref((double)x[i]));
        max_error = std::max(max_error, err);
        mean_error += err;
    }
    mean_error /= VEC_LEN;

    std::cout << "max_error=" << max_error << std::endl;
    std::cout << "mean_error=" << mean_error << std::endl;
    std::cout << (max_error < 0.25 ? "PASS" : "FAIL") << std::endl;
    return max_error < 0.25 ? 0 : 1;
}
