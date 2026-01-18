#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "rope: out and in must be 3D tensors");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D tensor");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be I64 type");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");
    ASSERT(out->shape() == in->shape(), "rope: out and in must have same shape");
    
    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];
    
    ASSERT(head_dim % 2 == 0, "rope: head_dim must be even");
    ASSERT(pos_ids->numel() == seq_len, "rope: pos_ids size must match seq_len");
    ASSERT(out->dtype() == in->dtype(), "rope: out and in must have same dtype");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        out->dtype(), seq_len, n_heads, head_dim);
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
