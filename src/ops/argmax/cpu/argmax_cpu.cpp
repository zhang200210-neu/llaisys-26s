#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <type_traits>

namespace {

// 实现 argmax：在 vals 中寻找最大值的索引和值，结果写入 max_idx / max_val
template <typename T>
void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    // 使用 float 比较以避免 fp16 / bf16 的精度问题
    using value_t = T;
    const auto *v = reinterpret_cast<const value_t *>(vals);
    auto *out_idx  = reinterpret_cast<int64_t *>(max_idx);
    auto *out_val  = reinterpret_cast<value_t *>(max_val);

    float best_value = llaisys::utils::cast<float>(v[0]);
    int64_t best_index = 0;

    for (size_t i = 1; i < numel; ++i) {
        const float cur = llaisys::utils::cast<float>(v[i]);
        if (cur > best_value) {
            best_value = cur;
            best_index = static_cast<int64_t>(i);
        }
    }

    *out_idx = best_index;
    *out_val = llaisys::utils::cast<value_t>(best_value);
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_impl<float>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
