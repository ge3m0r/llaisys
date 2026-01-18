#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    
    for (size_t i = 0; i < seq_len; i++) {
        int64_t pos = pos_ids[i];
        
        for (size_t h = 0; h < n_heads; h++) {
            const T *in_vec = in + (i * n_heads + h) * head_dim;
            T *out_vec = out + (i * n_heads + h) * head_dim;
            
            // Split into [a, b] pairs
            const T *a = in_vec;
            const T *b = in_vec + half_dim;
            T *a_out = out_vec;
            T *b_out = out_vec + half_dim;
            
            for (size_t j = 0; j < half_dim; j++) {
                // Calculate angle: phi = pos / (theta^(2j/d))
                float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
                float freq = std::pow(theta, exponent);
                float phi = static_cast<float>(pos) / freq;
                
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // Convert to float for computation
                float a_val = llaisys::utils::cast<float>(a[j]);
                float b_val = llaisys::utils::cast<float>(b[j]);
                
                // Apply rotation
                float a_out_val = a_val * cos_phi - b_val * sin_phi;
                float b_out_val = b_val * cos_phi + a_val * sin_phi;
                
                a_out[j] = llaisys::utils::cast<T>(a_out_val);
                b_out[j] = llaisys::utils::cast<T>(b_out_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                    reinterpret_cast<const llaisys::bf16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                    reinterpret_cast<const llaisys::fp16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
