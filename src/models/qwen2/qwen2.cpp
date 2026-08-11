#include "rearrange_cpu.hpp"

#include <cstring>

namespace {

void rearrange_recursive(std::byte *const out,
                         const std::byte *const in,
                         const std::vector<size_t> &shape,
                         const std::vector<ptrdiff_t> &out_strides,
                         const std::vector<ptrdiff_t> &in_strides,
                         const size_t elem_size,
                         const size_t dim,
                         const ptrdiff_t out_off,
                         const ptrdiff_t in_off) {
    if (dim == shape.size()) {
        // 到达叶子维度：拷贝单个元素
        std::memcpy(out + out_off * elem_size, in + in_off * elem_size, elem_size);
        return;
    }

    const size_t len   = shape[dim];
    const ptrdiff_t os = out_strides[dim];
    const ptrdiff_t is = in_strides[dim];

    for (size_t i = 0; i < len; ++i) {
        rearrange_recursive(out, in, shape, out_strides, in_strides,
                            elem_size, dim + 1,
                            out_off + i * os,
                            in_off  + i * is);
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void rearrange(std::byte *const out,
               const std::byte *const in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               const size_t elem_size) {
    rearrange_recursive(out, in, shape, out_strides, in_strides, elem_size, 0, 0, 0);
}

} // namespace llaisys::ops::cpu
