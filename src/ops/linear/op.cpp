#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    ASSERT(weight->ndim() == 2, "linear: weight must be a 2D tensor");
    ASSERT(in->isContiguous() && out->isContiguous() && weight->isContiguous(), "linear: all tensors must be contiguous");

    const size_t in_ndim = in->ndim();
    const size_t out_ndim = out->ndim();

    ASSERT(in_ndim == 2 || in_ndim == 3, "linear: in must be a 2D or 3D tensor");

    // Get features K and N from weight
    const size_t out_features = weight->shape()[0];
    const size_t in_features = weight->shape()[1];

    ASSERT(in->shape()[in_ndim - 1] == in_features, "linear: in_features mismatch");
    ASSERT(out->shape()[out_ndim - 1] == out_features, "linear: out_features mismatch");

    size_t batch_size = 1;
    if (in_ndim == 2) {
        batch_size = in->shape()[0];
        ASSERT(out_ndim == 2, "linear: out must be 2D if in is 2D");
        ASSERT(out->shape()[0] == batch_size, "linear: batch size mismatch");
    } else { // in_ndim == 3
        batch_size = in->shape()[0] * in->shape()[1];
        if (out_ndim == 3) {
            ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == out->shape()[1], "linear: batch/seq dims mismatch for 3D out");
        } else { // out_ndim == 2
            ASSERT(out->shape()[0] == batch_size, "linear: batch size mismatch for 2D out");
        }
    }

    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "linear: all tensors must have same dtype");

    if (bias) {
        ASSERT(bias->ndim() == 1, "linear: bias must be 1D tensor");
        ASSERT(bias->shape()[0] == out_features, "linear: bias size must match out_features");
        ASSERT(bias->dtype() == out->dtype(), "linear: bias must have same dtype");
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    const std::byte *bias_ptr = bias ? bias->data() : nullptr;

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                          out->dtype(), batch_size, in_features, out_features);
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
