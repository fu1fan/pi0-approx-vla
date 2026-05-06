#include "kernel.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

static gemm_i8_t input[GEMM_M][GEMM_K];
static gemm_i8_t weight[GEMM_K][GEMM_N];
static gemm_acc_t bias[GEMM_N];
static gemm_o16_t output[GEMM_M][GEMM_N];
static gemm_o16_t golden[GEMM_M][GEMM_N];

static gemm_o16_t ref_requantize(gemm_acc_t acc, int32_t scale_q15, int32_t shift) {
    int64_t scaled = static_cast<int64_t>(acc) * static_cast<int64_t>(scale_q15);
    if (shift > 0) {
        const int64_t rounding = static_cast<int64_t>(1) << (shift - 1);
        if (scaled >= 0) {
            scaled = (scaled + rounding) >> shift;
        } else {
            scaled = -(((-scaled) + rounding) >> shift);
        }
    }
    if (scaled > 32767) {
        return 32767;
    }
    if (scaled < -32768) {
        return -32768;
    }
    return static_cast<gemm_o16_t>(scaled);
}

static void init_data() {
    for (int m = 0; m < GEMM_M; ++m) {
        for (int k = 0; k < GEMM_K; ++k) {
            input[m][k] = static_cast<gemm_i8_t>(((m * 3 + k * 5 + 13) % 127) - 63);
        }
    }
    for (int k = 0; k < GEMM_K; ++k) {
        for (int n = 0; n < GEMM_N; ++n) {
            weight[k][n] = static_cast<gemm_i8_t>(((k * 7 + n * 11 + 19) % 127) - 63);
        }
    }
    for (int n = 0; n < GEMM_N; ++n) {
        bias[n] = static_cast<gemm_acc_t>((n % 31) - 15);
    }
}

static void compute_golden(int32_t scale_q15, int32_t shift) {
    for (int m = 0; m < GEMM_M; ++m) {
        for (int n = 0; n < GEMM_N; ++n) {
            gemm_acc_t acc = bias[n];
            for (int k = 0; k < GEMM_K; ++k) {
                acc += static_cast<gemm_acc_t>(input[m][k]) *
                       static_cast<gemm_acc_t>(weight[k][n]);
            }
            golden[m][n] = ref_requantize(acc, scale_q15, shift);
        }
    }
}

int main() {
    const int32_t scale_q15 = 1;
    const int32_t shift = 8;
    init_data();
    compute_golden(scale_q15, shift);

    int8_gemm_kernel(input, weight, bias, output, scale_q15, shift);

    double mse = 0.0;
    double mae = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_out = 0.0;
    double diff_norm = 0.0;
    int mismatches = 0;
    const double count = static_cast<double>(GEMM_M) * static_cast<double>(GEMM_N);

    for (int m = 0; m < GEMM_M; ++m) {
        for (int n = 0; n < GEMM_N; ++n) {
            const double ref = static_cast<double>(golden[m][n]);
            const double got = static_cast<double>(output[m][n]);
            const double diff = got - ref;
            if (golden[m][n] != output[m][n]) {
                ++mismatches;
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
    const double cosine = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + 1.0e-12);
    const double rel_l2 = std::sqrt(diff_norm) / (std::sqrt(norm_ref) + 1.0e-12);

    std::printf(
        "HLS_METRIC kernel=int8_gemm dtype=int8_acc32_out16 shape=%dx%dx%d mse=%.12e mae=%.12e cosine=%.12f relative_l2=%.12e mismatches=%d\n",
        GEMM_M,
        GEMM_K,
        GEMM_N,
        mse,
        mae,
        cosine,
        rel_l2,
        mismatches);
    return mismatches == 0 ? 0 : 1;
}
