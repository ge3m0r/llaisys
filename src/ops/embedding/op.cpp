#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->ndim() == 1, "embedding: index must be 1D tensor");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D tensor");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D tensor");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be I64 type");
    ASSERT(out->dtype() == weight->dtype(), "embedding: out and weight must have same dtype");
    ASSERT(out->shape()[0] == index->numel(), "embedding: out first dimension must match index size");
    ASSERT(out->shape()[1] == weight->shape()[1], "embedding: out second dimension must match weight embedding dimension");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
            "embedding: all tensors must be contiguous");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape()[1];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                             out->dtype(), num_indices, embedding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
