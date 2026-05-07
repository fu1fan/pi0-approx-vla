#include "kernel.hpp"

#include <cmath>
#include <cstdio>

static softmax_exact_in_t input[SOFTMAX_ROWS][SOFTMAX_LEN];
static softmax_exact_prob_t output[SOFTMAX_ROWS][SOFTMAX_LEN];
static double golden[SOFTMAX_ROWS][SOFTMAX_LEN];

static void init_data() {
    for (int r = 0; r < SOFTMAX_ROWS; ++r) {
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
            const double value = 0.03125 * static_cast<double>((r * 17 + c * 13) % 257) - 4.0;
            input[r][c] = static_cast<softmax_exact_in_t>(value);
        }
    }
}

static void compute_golden() {
    for (int r = 0; r < SOFTMAX_ROWS; ++r) {
        double max_value = static_cast<double>(input[r][0]);
        for (int c = 1; c < SOFTMAX_LEN; ++c) {
            const double value = static_cast<double>(input[r][c]);
            if (value > max_value) {
                max_value = value;
            }
        }

        double sum = 0.0;
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
            golden[r][c] = std::exp(static_cast<double>(input[r][c]) - max_value);
            sum += golden[r][c];
        }
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
            golden[r][c] /= sum;
        }
    }
}

int main() {
    init_data();
    compute_golden();
    exact_softmax_kernel(input, output);

    double mse = 0.0;
    double mae = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_out = 0.0;
    double diff_norm = 0.0;
    double kl = 0.0;
    double row_sum_max_abs_err = 0.0;
    int non_finite = 0;
    const double eps = 1.0e-12;
    const double count = static_cast<double>(SOFTMAX_ROWS) * static_cast<double>(SOFTMAX_LEN);

    for (int r = 0; r < SOFTMAX_ROWS; ++r) {
        double row_sum = 0.0;
        for (int c = 0; c < SOFTMAX_LEN; ++c) {
            const double ref = golden[r][c];
            const double got = static_cast<double>(output[r][c]);
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
            kl += ref * std::log((ref + eps) / (got + eps));
            row_sum += got;
        }
        const double sum_abs_err = std::fabs(row_sum - 1.0);
        if (sum_abs_err > row_sum_max_abs_err) {
            row_sum_max_abs_err = sum_abs_err;
        }
    }

    mse /= count;
    mae /= count;
    kl /= static_cast<double>(SOFTMAX_ROWS);
    const double cosine = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + eps);
    const double rel_l2 = std::sqrt(diff_norm) / (std::sqrt(norm_ref) + eps);

    std::printf(
        "HLS_METRIC kernel=exact_softmax variant=exp dtype=float32 shape=rows%d_len%d mse=%.12e mae=%.12e kl=%.12e cosine=%.12f relative_l2=%.12e row_sum_max_abs_err=%.12e non_finite=%d\n",
        SOFTMAX_ROWS,
        SOFTMAX_LEN,
        mse,
        mae,
        kl,
        cosine,
        rel_l2,
        row_sum_max_abs_err,
        non_finite);
    return non_finite == 0 ? 0 : 1;
}
