#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>

namespace {
    template <typename T>
    void linear_impl(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                     size_t m, size_t n, size_t k) {
        const auto *in_ptr   = reinterpret_cast<const T *>(in);
        const auto *w_ptr    = reinterpret_cast<const T *>(weight);
        const auto *bias_ptr = bias ? reinterpret_cast<const T *>(bias) : nullptr;
        auto *out_ptr        = reinterpret_cast<T *>(out);

        for (size_t i = 0; i < m; ++i) {
            const T *in_row = in_ptr + i * k;
            for (size_t o = 0; o < n; ++o) {
                // Accumulate bias if present
                float acc = bias_ptr ? llaisys::utils::cast<float>(bias_ptr[o]) : 0.0f;
                const T *w_row = w_ptr + o * k;  // weight row for output o
                // Dot product
                for (size_t j = 0; j < k; ++j) {
                    acc += llaisys::utils::cast<float>(in_row[j]) *
                           llaisys::utils::cast<float>(w_row[j]);
                }
                out_ptr[i * n + o] = llaisys::utils::cast<T>(acc);
            }
        }
    }
} // anonymous namespace

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(out, in, weight, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
