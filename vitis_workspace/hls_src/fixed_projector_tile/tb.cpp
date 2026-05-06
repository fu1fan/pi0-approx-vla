#include "kernel.hpp"

#include <cmath>
#include <cstdio>

static proj_t input[PROJ_TOKENS][PROJ_IN_DIM];
static proj_t weight[PROJ_IN_DIM][PROJ_OUT_DIM];
static proj_t bias[PROJ_OUT_DIM];
static proj_t output[PROJ_TOKENS][PROJ_OUT_DIM];
static double golden[PROJ_TOKENS][PROJ_OUT_DIM];

static void init_data() {
    for (int t = 0; t < PROJ_TOKENS; ++t) {
        for (int i = 0; i < PROJ_IN_DIM; ++i) {
            const double value = 0.001 * static_cast<double>((t * 31 + i * 17) % 257) - 0.128;
            input[t][i] = static_cast<proj_t>(value);
        }
    }
    for (int i = 0; i < PROJ_IN_DIM; ++i) {
        for (int o = 0; o < PROJ_OUT_DIM; ++o) {
            const double value = 0.0005 * static_cast<double>((i * 13 + o * 19) % 257) - 0.064;
            weight[i][o] = static_cast<proj_t>(value);
        }
    }
    for (int o = 0; o < PROJ_OUT_DIM; ++o) {
        bias[o] = static_cast<proj_t>(0.00025 * static_cast<double>((o * 23) % 257) - 0.032);
    }
}

static void compute_golden() {
    for (int t = 0; t < PROJ_TOKENS; ++t) {
        for (int o = 0; o < PROJ_OUT_DIM; ++o) {
            double acc = static_cast<double>(bias[o]);
            for (int i = 0; i < PROJ_IN_DIM; ++i) {
                acc += static_cast<double>(input[t][i]) * static_cast<double>(weight[i][o]);
            }
            golden[t][o] = acc;
        }
    }
}

int main() {
    init_data();
    compute_golden();
    fixed_projector_tile_kernel(input, weight, bias, output);

    double mse = 0.0;
    double mae = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_out = 0.0;
    double diff_norm = 0.0;
    int non_finite = 0;
    const double eps = 1.0e-12;
    const double count = static_cast<double>(PROJ_TOKENS) * static_cast<double>(PROJ_OUT_DIM);

    for (int t = 0; t < PROJ_TOKENS; ++t) {
        for (int o = 0; o < PROJ_OUT_DIM; ++o) {
            const double ref = golden[t][o];
            const double got = static_cast<double>(output[t][o]);
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
    }

    mse /= count;
    mae /= count;
    const double cosine = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + eps);
    const double rel_l2 = std::sqrt(diff_norm) / (std::sqrt(norm_ref) + eps);

    std::printf(
        "HLS_METRIC kernel=fixed_projector_tile dtype=fixed16x6_acc40x16 shape=%dx%dx%d mse=%.12e mae=%.12e cosine=%.12f relative_l2=%.12e non_finite=%d\n",
        PROJ_TOKENS,
        PROJ_IN_DIM,
        PROJ_OUT_DIM,
        mse,
        mae,
        cosine,
        rel_l2,
        non_finite);
    return non_finite == 0 ? 0 : 1;
}
