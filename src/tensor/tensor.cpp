#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

// 构造器
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

// 创建一个新的张量
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    // 计算步长
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    // 后面所有维长度的乘积
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    // 计算数据类型大小
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

// 返回指向张量数据的指针
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

// 返回指向张量数据的常量指针
const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

// 返回张量的维度数
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

// 返回张量的形状
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

// 返回张量的步长
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

// 返回张量的数据类型
llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

// 返回张量所存储数据的存储对象
llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

// 返回张量所在设备的ID
int Tensor::deviceId() const {
    return _storage->deviceId();
}

// 返回张量中的元素数量
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

// 返回张量中每个元素的大小（以字节为单位）
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

// 调试信息
std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();
    return ss.str();
}

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
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
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

// 任务 1.2：检查张量是否连续
bool Tensor::isContiguous() const {
    const auto &sh = shape();
    const auto &st = strides();
    if (sh.empty()) return true;

    size_t expect = 1;
    for (size_t i = sh.size(); i-- > 0;) {
        if (sh[i] == 1) continue;       // 长度为 1 的维可跳过
        if (st[i] != static_cast<ptrdiff_t>(expect)) {
            return false;
        }
        expect *= sh[i];
    }
    return true;
}

// 任务 1.4：permute（移除多余 return）
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
    return tensor_t(new Tensor(new_meta, _storage, _offset));   // 零拷贝
}

// 辅助函数：获得连续存储的张量（修正连续分支丢失 offset 的问题）
tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return tensor_t(new Tensor(_meta, _storage, _offset));  // 保留原始偏移
    }

    const auto &sh = shape();
    const auto dim = sh.size();
    std::vector<ptrdiff_t> c_str(dim, 1);
    for (size_t i = dim - 1; i-- > 0;) {
        c_str[i] = c_str[i + 1] * sh[i + 1];
    }

    size_t bytes = numel() * elementSize();
    core::storage_t st = (deviceType() == LLAISYS_DEVICE_CPU)
                         ? core::context().runtime().allocateHostStorage(bytes)
                         : core::context().runtime().allocateDeviceStorage(bytes);

    tensor_t dst(new Tensor(TensorMeta{dtype(), sh, c_str}, st, 0));

    core::context().setDevice(deviceType(), deviceId());
    core::context().runtime().api()->memcpy_sync(
        dst->data(), data(), bytes,
        deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);

    return dst;
}

// 任务 1.3：view（修复内存泄漏，直接共享存储）
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::invalid_argument("view: total elements mismatch");
    }

    if (isContiguous()) {
        // 连续时直接计算新步长，共享同一块存储和偏移
        std::vector<ptrdiff_t> new_strides(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= shape[i];
        }
        TensorMeta new_meta{dtype(), shape, new_strides};
        return tensor_t(new Tensor(new_meta, _storage, _offset));
    } else {
        // 非连续时先转换为连续，再调用 view
        return contiguous()->view(shape);
    }
}

// 任务 1.5：slice
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

// 任务 1.1：load（从主机内存加载数据到张量）
void Tensor::load(const void *src_) {
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

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
