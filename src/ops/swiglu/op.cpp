#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    
    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2,
           "swiglu: all tensors must be 2D");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");
    ASSERT(out->shape() == gate->shape() && gate->shape() == up->shape(),
           "swiglu: all tensors must have same shape");
    ASSERT(out->dtype() == gate->dtype() && gate->dtype() == up->dtype(),
           "swiglu: all tensors must have same dtype");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    size_t numel = out->numel();

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                          out->dtype(), numel);
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
