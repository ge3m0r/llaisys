#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T max_value = vals[0];
    size_t max_index = 0;
    
    for (size_t i = 1; i < numel; i++) {
        T val = vals[i];
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float val_f = llaisys::utils::cast<float>(val);
            float max_f = llaisys::utils::cast<float>(max_value);
            if (val_f > max_f) {
                max_value = val;
                max_index = i;
            }
        } else {
            if (val > max_value) {
                max_value = val;
                max_index = i;
            }
        }
    }
    
    *max_idx = static_cast<int64_t>(max_index);
    *max_val = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t val_type, llaisysDataType_t idx_type, size_t numel) {
    ASSERT(idx_type == LLAISYS_DTYPE_I64, "argmax: max_idx must be I64 type");
    
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val), 
                      reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
