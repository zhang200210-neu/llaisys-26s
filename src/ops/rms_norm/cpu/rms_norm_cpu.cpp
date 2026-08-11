#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
    template <typename T>
    void rms_norm_impl(std::byte *const out,
                       const std::byte *const in,
                       const std::byte *const weight,
                       const size_t rows,
                       const size_t cols,
                       const float eps) {
        const auto *in_ptr   = reinterpret_cast<const T *>(in);
        const auto *w_ptr    = reinterpret_cast<const T *>(weight);
        auto       *out_ptr  = reinterpret_cast<T *>(out);

        const float cols_f = static_cast<float>(cols);

        for (size_t i = 0; i < rows; ++i) {
            const T *row_in  = in_ptr  + i * cols;
            T       *row_out = out_ptr + i * cols;

            // Compute mean of squares
            float sum_sq = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                const float v = llaisys::utils::cast<float>(row_in[j]);
                sum_sq += v * v;
            }
            const float mean   = sum_sq / cols_f;
            const float inv_rms = 1.0f / std::sqrt(mean + eps);

            // Apply normalization and element‑wise scaling
            for (size_t j = 0; j < cols; ++j) {
                const float v = llaisys::utils::cast<float>(row_in[j]);
                const float w = llaisys::utils::cast<float>(w_ptr[j]);
                row_out[j] = llaisys::utils::cast<T>(v * inv_rms * w);
            }
        }
    }
} // anonymous namespace

namespace llaisys::ops::cpu {
void rms_norm(std::byte *const out,
              const std::byte *const in,
              const std::byte *const weight,
              const llaisysDataType_t type,
              const size_t rows,
              const size_t cols,
              const float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
