#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "self_attention: all tensors must be 3D");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");
    
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "self_attention: attn_val shape must match q shape");
    ASSERT(k->shape()[2] == hd && v->shape()[2] == hd,
           "self_attention: k and v head_dim must match q head_dim");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh,
           "self_attention: v shape must match k shape");
    ASSERT(nh % nkvh == 0, "self_attention: nh must be divisible by nkvh (GQA)");
    ASSERT(attn_val->dtype() == q->dtype() && q->dtype() == k->dtype() && k->dtype() == v->dtype(),
           "self_attention: all tensors must have same dtype");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                  attn_val->dtype(), qlen, kvlen, nh, nkvh, hd);
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
