#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

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

bool Tensor::isContiguous() const {
    // A tensor is contiguous if strides[i] = strides[i+1] * shape[i+1] for all i < ndim-1
    // and strides[ndim-1] = 1
    size_t ndim_ = this->ndim();
    if (ndim_ == 0) {
        return true;
    }
    
    // Check if the last stride is 1
    if (_meta.strides[ndim_ - 1] != 1) {
        return false;
    }
    
    // Check if each stride is equal to the next stride times the next dimension size
    for (size_t i = 0; i < ndim_ - 1; i++) {
        if (_meta.strides[i] != _meta.strides[i + 1] * _meta.shape[i + 1]) {
            return false;
        }
    }
    
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    CHECK_ARGUMENT(order.size() == ndim_, "permute order size must match tensor ndim");
    
    // Validate order indices
    std::vector<bool> used(ndim_, false);
    for (size_t idx : order) {
        CHECK_ARGUMENT(idx < ndim_, "permute order index out of range");
        CHECK_ARGUMENT(!used[idx], "permute order contains duplicate indices");
        used[idx] = true;
    }
    
    // Create new shape and strides according to the order
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    
    for (size_t i = 0; i < ndim_; i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // Calculate total elements in new shape (excluding inferred dimension)
    size_t new_numel = 1;
    size_t inferred_dim = std::numeric_limits<size_t>::max();
    
    for (size_t i = 0; i < shape.size(); i++) {
        // Check for inferred dimension: using a large sentinel value
        // In practice, this would be passed as -1 from Python, which gets converted
        // For now, we'll check for the max value as a sentinel
        if (shape[i] == std::numeric_limits<size_t>::max()) {
            CHECK_ARGUMENT(inferred_dim == std::numeric_limits<size_t>::max(), 
                          "view can only have one inferred dimension (-1)");
            inferred_dim = i;
        } else {
            new_numel *= shape[i];
        }
    }
    
    size_t old_numel = this->numel();
    
    // Calculate the actual shape with inferred dimension resolved
    std::vector<size_t> new_shape_fixed = shape;
    if (inferred_dim != std::numeric_limits<size_t>::max()) {
        CHECK_ARGUMENT(new_numel > 0 && old_numel % new_numel == 0, 
                      "view shape is not compatible with tensor size");
        new_shape_fixed[inferred_dim] = old_numel / new_numel;
        new_numel = old_numel;  // Update to total elements
    }
    
    // Check if total number of elements matches
    CHECK_ARGUMENT(new_numel == old_numel, 
                  "view shape must have same number of elements as original tensor");
    
    // For view to work, the tensor must be contiguous
    CHECK_ARGUMENT(this->isContiguous(), "view requires a contiguous tensor");
    
    // Calculate strides for the new shape (contiguous layout)
    std::vector<ptrdiff_t> new_strides(new_shape_fixed.size());
    size_t stride = 1;
    for (size_t i = 1; i <= new_shape_fixed.size(); i++) {
        size_t dim_idx = new_shape_fixed.size() - i;
        new_strides[dim_idx] = stride;
        stride *= new_shape_fixed[dim_idx];
    }
    
    TensorMeta new_meta{_meta.dtype, new_shape_fixed, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim_ = this->ndim();
    CHECK_ARGUMENT(dim < ndim_, "slice dimension index out of range");
    CHECK_ARGUMENT(start <= end, "slice start must be <= end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end out of range");
    
    // Calculate new shape and strides
    std::vector<size_t> new_shape = _meta.shape;
    std::vector<ptrdiff_t> new_strides = _meta.strides;
    
    // Update shape for the sliced dimension
    new_shape[dim] = end - start;
    
    // Calculate new offset: add start * stride[dim] * element_size to the current offset
    // strides are in elements, so we need to convert to bytes
    size_t element_size = this->elementSize();
    size_t new_offset = _offset + static_cast<size_t>(start * _meta.strides[dim]) * element_size;
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    size_t total_bytes = this->numel() * this->elementSize();
    
    // Determine memcpy kind based on device type
    llaisysMemcpyKind_t kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else {
        kind = LLAISYS_MEMCPY_H2D;
    }
    
    // Perform the memory copy
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        total_bytes,
        kind
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
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
