#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
    template <typename T>
    void rope_impl(std::byte *const out,
                   const std::byte *const in,
                   const std::byte *const pos_ids,
                   const size_t seqlen,
                   const size_t nhead,
                   const size_t dim,
                   const float theta) {
        const auto *in_data  = reinterpret_cast<const T *>(in);
        const auto *pos_data = reinterpret_cast<const int64_t *>(pos_ids);
        auto       *out_data = reinterpret_cast<T *>(out);

        const size_t head_stride = dim;
        const size_t seq_stride  = nhead * dim;
        const size_t half_dim    = dim / 2;

        for (size_t s = 0; s < seqlen; ++s) {
            const float position = static_cast<float>(pos_data[s]);

            for (size_t h = 0; h < nhead; ++h) {
                const T *x = in_data  + s * seq_stride + h * head_stride;
                T       *y = out_data + s * seq_stride + h * head_stride;

                for (size_t j = 0; j < half_dim; ++j) {
                    // RoPE frequency
                    const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(dim);
                    const float angle = position / std::pow(theta, exponent);
                    const float sin_val = std::sin(angle);
                    const float cos_val = std::cos(angle);

                    const float a = llaisys::utils::cast<float>(x[j]);
                    const float b = llaisys::utils::cast<float>(x[half_dim + j]);

                    y[j]            = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    y[half_dim + j] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                }
            }
        }
    }
} // anonymous namespace

namespace llaisys::ops::cpu {
void rope(std::byte *const out,
          const std::byte *const in,
          const std::byte *const pos_ids,
          const llaisysDataType_t type,
          const size_t seqlen,
          const size_t nhead,
          const size_t dim,
          const float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
