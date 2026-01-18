#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
               size_t batch_size, size_t dim) {
    for (size_t i = 0; i < batch_size; i++) {
        const T *in_row = in + i * dim;
        T *out_row = out + i * dim;
        const T *w = weight;
        
        // Calculate RMS: sqrt((1/d) * sum(x_j^2) + eps)
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            float val = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += val * val;
        }
        
        float mean_sq = sum_sq / static_cast<float>(dim);
        float rms = std::sqrt(mean_sq + eps);
        float rms_inv = 1.0f / rms;
        
        // Apply normalization: Y_i = (W_i * X_i) / rms
        for (size_t j = 0; j < dim; j++) {
            float x_val = llaisys::utils::cast<float>(in_row[j]);
            float w_val = llaisys::utils::cast<float>(w[j]);
            float result = (w_val * x_val) * rms_inv;
            out_row[j] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t dtype, size_t batch_size, size_t dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight),
                        eps, batch_size, dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                        eps, batch_size, dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                        eps, batch_size, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
