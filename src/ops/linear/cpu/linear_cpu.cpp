#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    // Y = xW^T + b
    // out: [batch_size, out_features]
    // in: [batch_size, in_features]
    // weight: [out_features, in_features] (not transposed)
    // bias: [out_features] (optional)
    
    for (size_t i = 0; i < batch_size; i++) {
        const T *in_row = in + i * in_features;
        T *out_row = out + i * out_features;
        
        // Initialize output row with bias or zero
        if (bias) {
            for (size_t j = 0; j < out_features; j++) {
                out_row[j] = bias[j];
            }
        } else {
            for (size_t j = 0; j < out_features; j++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_row[j] = llaisys::utils::cast<T>(0.0f);
                } else {
                    out_row[j] = T(0);
                }
            }
        }
        
        // Matrix multiplication: out_row += in_row @ weight^T
        for (size_t j = 0; j < out_features; j++) {
            const T *weight_row = weight + j * in_features;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // For fp16/bf16, accumulate in float32 to avoid precision loss
                float sum_f = 0.0f;
                for (size_t k = 0; k < in_features; k++) {
                    float in_val = llaisys::utils::cast<float>(in_row[k]);
                    float w_val = llaisys::utils::cast<float>(weight_row[k]);
                    sum_f += in_val * w_val;
                }
                float out_val = llaisys::utils::cast<float>(out_row[j]);
                out_row[j] = llaisys::utils::cast<T>(out_val + sum_f);
            } else {
                // For f32, accumulate directly
                T sum = T(0);
                for (size_t k = 0; k < in_features; k++) {
                    sum += in_row[k] * weight_row[k];
                }
                out_row[j] += sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                      reinterpret_cast<const float *>(in),
                      reinterpret_cast<const float *>(weight),
                      bias ? reinterpret_cast<const float *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(in),
                      reinterpret_cast<const llaisys::bf16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(in),
                      reinterpret_cast<const llaisys::fp16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
