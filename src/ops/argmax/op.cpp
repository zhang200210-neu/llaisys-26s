#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
void argmax_cpu(int64_t *max_idx, T *max_val, const T *vals, size_t count) {
    size_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < count; ++i) {
        const float value = llaisys::utils::cast<float>(vals[i]);
        if (value > best_val) {
            best_idx = i;
            best_val = value;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = vals[best_idx];
}
} // namespace

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->deviceType() == LLAISYS_DEVICE_CPU, "Argmax currently supports CPU tensors only");
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "Argmax input must be a non-empty 1D tensor");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1, "Argmax index output must contain one element");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1, "Argmax value output must contain one element");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must use int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
                   "Argmax tensors must be contiguous");

    auto *idx = reinterpret_cast<int64_t *>(max_idx->data());
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_cpu(idx, reinterpret_cast<float *>(max_val->data()),
                          reinterpret_cast<const float *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_F16:
        return argmax_cpu(idx, reinterpret_cast<fp16_t *>(max_val->data()),
                          reinterpret_cast<const fp16_t *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_BF16:
        return argmax_cpu(idx, reinterpret_cast<bf16_t *>(max_val->data()),
                          reinterpret_cast<const bf16_t *>(vals->data()), vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
