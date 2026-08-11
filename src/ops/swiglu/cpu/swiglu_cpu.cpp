#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
    template <typename T>
    void swiglu_impl(std::byte *const out,
                     const std::byte *const gate,
                     const std::byte *const up,
                     const size_t numel) {
        const auto *g_ptr = reinterpret_cast<const T *>(gate);
        const auto *u_ptr = reinterpret_cast<const T *>(up);
        auto       *o_ptr = reinterpret_cast<T *>(out);

        for (size_t i = 0; i < numel; ++i) {
            const float g = llaisys::utils::cast<float>(g_ptr[i]);
            const float u = llaisys::utils::cast<float>(u_ptr[i]);
            const float sigmoid = 1.0f / (1.0f + std::exp(-g));
            o_ptr[i] = llaisys::utils::cast<T>(u * g * sigmoid);
        }
    }
} // anonymous namespace

namespace llaisys::ops::cpu {
void swiglu(std::byte *const out,
            const std::byte *const gate,
            const std::byte *const up,
            const llaisysDataType_t type,
            const size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
