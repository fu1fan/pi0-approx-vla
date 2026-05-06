#include "kernel.hpp"

#include <cmath>
#include <cstdio>

static gelu_t input[GELU_LEN];
static gelu_t output[GELU_LEN];
static double golden[GELU_LEN];

static double exact_gelu(double x) {
    const double kAlpha = 0.7978845608028654;
    return 0.5 * x * (1.0 + std::tanh(kAlpha * (x + 0.044715 * x * x * x)));
}

static void init_data() {
    for (int i = 0; i < GELU_LEN; ++i) {
        const double value = 0.0025 * static_cast<double>((i * 37) % 4001) - 5.0;
        input[i] = static_cast<gelu_t>(value);
        golden[i] = exact_gelu(static_cast<double>(input[i]));
    }
}

int main() {
    init_data();
    gelu_pwl_kernel(input, output);

    double mse = 0.0;
    double mae = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_out = 0.0;
    double diff_norm = 0.0;
    int non_finite = 0;
    const double eps = 1.0e-12;

    for (int i = 0; i < GELU_LEN; ++i) {
        const double ref = golden[i];
        const double got = static_cast<double>(output[i]);
        const double diff = got - ref;
        if (!std::isfinite(got)) {
            ++non_finite;
        }
        mse += diff * diff;
        mae += std::fabs(diff);
        dot += got * ref;
        norm_ref += ref * ref;
        norm_out += got * got;
        diff_norm += diff * diff;
    }

    mse /= static_cast<double>(GELU_LEN);
    mae /= static_cast<double>(GELU_LEN);
    const double cosine = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + eps);
    const double rel_l2 = std::sqrt(diff_norm) / (std::sqrt(norm_ref) + eps);

    std::printf(
        "HLS_METRIC kernel=gelu_pwl dtype=fixed16x6 shape=len%d segments=16 range_min=-4 range_max=4 mse=%.12e mae=%.12e cosine=%.12f relative_l2=%.12e non_finite=%d\n",
        GELU_LEN,
        mse,
        mae,
        cosine,
        rel_l2,
        non_finite);
    return non_finite == 0 ? 0 : 1;
}
