#include "kernel.hpp"

#include <cmath>
#include <cstdio>

static rms_exact_t input[RMSNORM_HIDDEN];
static rms_exact_t weight[RMSNORM_HIDDEN];
static rms_exact_t output[RMSNORM_HIDDEN];
static double golden[RMSNORM_HIDDEN];

static void init_data() {
    double sum_sq = 0.0;
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
        const double x = 0.00125 * static_cast<double>((i * 29) % 3201) - 2.0;
        const double w = 0.9 + 0.00025 * static_cast<double>((i * 11) % 801);
        input[i] = static_cast<rms_exact_t>(x);
        weight[i] = static_cast<rms_exact_t>(w);
        sum_sq += static_cast<double>(input[i]) * static_cast<double>(input[i]);
    }
    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(RMSNORM_HIDDEN) + 1.0e-5);
    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
        golden[i] = static_cast<double>(input[i]) * static_cast<double>(weight[i]) * inv_rms;
    }
}

int main() {
    init_data();
    exact_rmsnorm_kernel(input, weight, output);

    double mse = 0.0;
    double mae = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_out = 0.0;
    double diff_norm = 0.0;
    int non_finite = 0;
    const double eps = 1.0e-12;

    for (int i = 0; i < RMSNORM_HIDDEN; ++i) {
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

    mse /= static_cast<double>(RMSNORM_HIDDEN);
    mae /= static_cast<double>(RMSNORM_HIDDEN);
    const double cosine = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + eps);
    const double rel_l2 = std::sqrt(diff_norm) / (std::sqrt(norm_ref) + eps);

    std::printf(
        "HLS_METRIC kernel=exact_rmsnorm variant=sqrt dtype=float32 shape=hidden%d mse=%.12e mae=%.12e cosine=%.12f relative_l2=%.12e non_finite=%d\n",
        RMSNORM_HIDDEN,
        mse,
        mae,
        cosine,
        rel_l2,
        non_finite);
    return non_finite == 0 ? 0 : 1;
}
