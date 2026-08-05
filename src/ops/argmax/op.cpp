#include "op.hpp"

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return argmax_impl<float>(max_idx, max_val, vals, numel);
	case LLAISYS_DTYPE_BF16:
		return argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
	case LLAISYS_DTYPE_F16:
		return argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}
} // namespace llaisys::ops::cpu
