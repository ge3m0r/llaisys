#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    // attn_val: [qlen, nh, hd]
    // q: [qlen, nh, hd]
    // k: [kvlen, nkvh, hd]
    // v: [kvlen, nkvh, hd]
    
    // Expand k and v to match nh if needed (GQA)
    size_t head_ratio = nh / nkvh;
    
    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nh; h++) {
            size_t kvh = h / head_ratio;  // Which kv head to use
            
            const T *q_vec = q + (i * nh + h) * hd;
            T *out_vec = attn_val + (i * nh + h) * hd;
            
            // Compute attention scores: A = Q * K^T * scale
            std::vector<float> attn_scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();
            
            for (size_t j = 0; j < kvlen; j++) {
                const T *k_vec = k + (j * nkvh + kvh) * hd;
                
                // Dot product
                float dot = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    float q_val = llaisys::utils::cast<float>(q_vec[d]);
                    float k_val = llaisys::utils::cast<float>(k_vec[d]);
                    dot += q_val * k_val;
                }
                
                float score = dot * scale;
                
                // Apply causal mask: tril(diagonal=S-L) means j <= i + (kvlen - qlen)
                // temp_mask[j][i] = True if j <= i + (S-L), else False
                // attn_bias is -inf where temp_mask is False
                // So score should be -inf if j > i + (kvlen - qlen)
                if (j > i + (kvlen - qlen)) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                attn_scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }
            
            // Compute softmax
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                float exp_val = std::exp(attn_scores[j] - max_score);
                attn_scores[j] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[j] /= sum_exp;
            }
            
            // Compute weighted sum of values
            for (size_t d = 0; d < hd; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    const T *v_vec = v + (j * nkvh + kvh) * hd;
                    float v_val = llaisys::utils::cast<float>(v_vec[d]);
                    sum += attn_scores[j] * v_val;
                }
                out_vec[d] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
