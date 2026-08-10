#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>
#include <functional>   // for std::function

namespace llaisys {


Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

// create 静态方法（未改动）
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

// data / ndim / shape / strides / dtype / deviceType / deviceId / numel / elementSize / info（未改动）
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) { ss << s << " "; }
    ss << "] strides[ ";
    for (auto s : this->strides()) { ss << s << " "; }
    ss << "] dtype=" << this->dtype();
    return ss.str();
}

// debug 辅助函数（未改动）
template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:   return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:   return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:     return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:    return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:    return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:    return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:     return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:    return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:    return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:    return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:    return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:    return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:    return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:   return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}


bool Tensor::isContiguous() const {
    const auto &sh = shape();
    const auto &st = strides();
    if (sh.empty()) return true;   
    
    size_t expect = 1;
    for (size_t i = sh.size(); i-- > 0;) {
        if (sh[i] == 1) continue;       // 长度为1的维度可以拥有任意步长
        if (st[i] != static_cast<ptrdiff_t>(expect)) {
            return false;
        }
        expect *= sh[i];
    }
    return true;
}


tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != ndim()) {
        throw std::invalid_argument("permute: order length mismatch");
    }

    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        size_t j = order[i];
        if (j >= ndim()) throw std::out_of_range("permute index");
        new_shape[i]   = shape()[j];
        new_strides[i] = strides()[j];
    }

    TensorMeta new_meta{dtype(), new_shape, new_strides};
    return tensor_t(new Tensor(new_meta, _storage, _offset));   // 零拷贝，只保留一个 return
}

// ------------------------------------------------------------
// 辅助函数：获得连续存储的张量（修复了非连续情况下内存拷贝错误的问题）
// ------------------------------------------------------------
tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::make_shared<Tensor>(_meta, _storage, _offset);
    }

    // 1. 申请连续存储
    const auto &sh = shape();
    size_t bytes = numel() * elementSize();
    core::storage_t st = (deviceType() == LLAISYS_DEVICE_CPU)
                         ? core::context().runtime().allocateHostStorage(bytes)
                         : core::context().runtime().allocateDeviceStorage(bytes);

    std::vector<ptrdiff_t> c_strides(sh.size());
    size_t stride = 1;
    for (size_t i = sh.size(); i-- > 0;) {
        c_strides[i] = stride;
        stride *= sh[i];
    }
    tensor_t dst = std::make_shared<Tensor>(TensorMeta{dtype(), sh, c_strides}, st, 0);

    // 3. 数据复制
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU 下使用递归按元素复制，正确处理非连续步长
        std::function<void(size_t, size_t, size_t)> copy_loop;
        copy_loop = [&](size_t dim, size_t src_off, size_t dst_off) {
            if (dim == ndim()) {
                std::memcpy(dst->data() + dst_off, data() + src_off, elementSize());
            } else {
                for (size_t i = 0; i < sh[dim]; ++i) {
                    copy_loop(dim + 1,
                              src_off + i * strides()[dim] * elementSize(),
                              dst_off + i * c_strides[dim] * elementSize());
                }
            }
        };
        copy_loop(0, 0, 0);
    } else {
        // 设备端非连续 -> 连续的重排需要自定义内核，此处暂未实现
        throw std::runtime_error("contiguous() on non-CPU device is not implemented yet");
    }

    return dst;
}


tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查元素总数是否匹配
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::invalid_argument("view: total elements mismatch");
    }

    if (isContiguous()) {
        // 连续情况下直接计算新步长，共享存储与偏移
        std::vector<ptrdiff_t> new_strides(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= shape[i];
        }
        TensorMeta new_meta{dtype(), shape, new_strides};
        return std::make_shared<Tensor>(new_meta, _storage, _offset);
    } else {
        // 非连续时，先变为连续再调用 view
        return contiguous()->view(shape);
    }
}


tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= ndim()) throw std::out_of_range("slice dim");
    if (start > end || end > shape()[dim])
        throw std::out_of_range("slice range");

    auto new_shape   = shape();
    auto new_strides = strides();
    new_shape[dim]   = end - start;

    size_t new_offset = _offset + start * new_strides[dim] * elementSize();

    TensorMeta new_meta{dtype(), new_shape, new_strides};
    return tensor_t(new Tensor(new_meta, _storage, new_offset));
}


void Tensor::load(const void *src_) {
    // 要求张量为连续存储，否则无法按连续块拷贝
    if (!isContiguous()) {
        throw std::runtime_error("load requires contiguous tensor");
    }
    size_t bytes = numel() * elementSize();
    std::byte *dst = data();

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst, src_, bytes);
    } else {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            dst, src_, bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

// reshape / to（保持原样）
tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
