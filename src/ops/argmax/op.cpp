#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->ndim() == 1, "argmax: vals must be 1D tensor");
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1, "argmax: max_idx must be 1D tensor with single element");
    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1, "argmax: max_val must be 1D tensor with single element");
    ASSERT(max_val->dtype() == vals->dtype(), "argmax: max_val must have same dtype as vals");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be I64 type");
    ASSERT(vals->isContiguous(), "argmax: vals must be contiguous");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                          vals->dtype(), max_idx->dtype(), vals->numel());
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
