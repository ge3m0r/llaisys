#include "qwen2.hpp"

#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/add/op.hpp"

#include "../../utils/check.hpp"

#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
#include <utility>

namespace llaisys {
namespace models {
namespace qwen2 {

Qwen2Model::Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device, int device_id)
    : config_(config) {

    size_t head_dim = config.hidden_size / config.n_head;

    // Create KV cache
    kv_cache_ = std::make_shared<KVCache>();
    kv_cache_->seq_len = 0;
    kv_cache_->keys.resize(config.n_layer);
    kv_cache_->values.resize(config.n_layer);

    for (size_t i = 0; i < config.n_layer; i++) {
        // Create KV cache tensors for each layer
        std::vector<size_t> kv_shape = {config.max_seq_len, config.n_kv_head, head_dim};
        kv_cache_->keys[i] = Tensor::create(kv_shape, config.dtype, device, device_id);
        kv_cache_->values[i] = Tensor::create(kv_shape, config.dtype, device, device_id);

        // Initialize KV cache with zeros
        size_t kv_bytes = config.max_seq_len * config.n_kv_head * head_dim * llaisys::utils::dsize(config.dtype);
        std::memset(kv_cache_->keys[i]->data(), 0, kv_bytes);
        std::memset(kv_cache_->values[i]->data(), 0, kv_bytes);
    }

    // Initialize weight vectors
    weights_temp_.attn_norm.resize(config.n_layer);
    weights_temp_.attn_q_proj.resize(config.n_layer);
    weights_temp_.attn_q_bias.resize(config.n_layer);
    weights_temp_.attn_k_proj.resize(config.n_layer);
    weights_temp_.attn_k_bias.resize(config.n_layer);
    weights_temp_.attn_v_proj.resize(config.n_layer);
    weights_temp_.attn_v_bias.resize(config.n_layer);
    weights_temp_.attn_o_proj.resize(config.n_layer);
    weights_temp_.ffn_norm.resize(config.n_layer);
    weights_temp_.ffn_gate_proj.resize(config.n_layer);
    weights_temp_.ffn_up_proj.resize(config.n_layer);
    weights_temp_.ffn_down_proj.resize(config.n_layer);
}

// Weight setters
void Qwen2Model::set_weights_field(const std::string& name, const tensor_t& tensor) {
    if (name == "embed_tokens") weights_temp_.embed_tokens = tensor;
    else if (name == "lm_head") weights_temp_.lm_head = tensor;
    else if (name == "norm_f") weights_temp_.norm_f = tensor;
}

void Qwen2Model::set_attn_norm(size_t layer, const tensor_t& tensor) { weights_temp_.attn_norm[layer] = tensor; }
void Qwen2Model::set_attn_q_proj(size_t layer, const tensor_t& tensor) { weights_temp_.attn_q_proj[layer] = tensor; }
void Qwen2Model::set_attn_q_bias(size_t layer, const tensor_t& tensor) { weights_temp_.attn_q_bias[layer] = tensor; }
void Qwen2Model::set_attn_k_proj(size_t layer, const tensor_t& tensor) { weights_temp_.attn_k_proj[layer] = tensor; }
void Qwen2Model::set_attn_k_bias(size_t layer, const tensor_t& tensor) { weights_temp_.attn_k_bias[layer] = tensor; }
void Qwen2Model::set_attn_v_proj(size_t layer, const tensor_t& tensor) { weights_temp_.attn_v_proj[layer] = tensor; }
void Qwen2Model::set_attn_v_bias(size_t layer, const tensor_t& tensor) { weights_temp_.attn_v_bias[layer] = tensor; }
void Qwen2Model::set_attn_o_proj(size_t layer, const tensor_t& tensor) { weights_temp_.attn_o_proj[layer] = tensor; }
void Qwen2Model::set_ffn_norm(size_t layer, const tensor_t& tensor) { weights_temp_.ffn_norm[layer] = tensor; }
void Qwen2Model::set_ffn_gate_proj(size_t layer, const tensor_t& tensor) { weights_temp_.ffn_gate_proj[layer] = tensor; }
void Qwen2Model::set_ffn_up_proj(size_t layer, const tensor_t& tensor) { weights_temp_.ffn_up_proj[layer] = tensor; }
void Qwen2Model::set_ffn_down_proj(size_t layer, const tensor_t& tensor) { weights_temp_.ffn_down_proj[layer] = tensor; }

void Qwen2Model::build() {
    // Copy temporary weights to built weights
    weights_.embed_tokens = weights_temp_.embed_tokens;
    weights_.lm_head = weights_temp_.lm_head;
    weights_.norm_f = weights_temp_.norm_f;
    weights_.attn_norm = weights_temp_.attn_norm;
    weights_.attn_q_proj = weights_temp_.attn_q_proj;
    weights_.attn_q_bias = weights_temp_.attn_q_bias;
    weights_.attn_k_proj = weights_temp_.attn_k_proj;
    weights_.attn_k_bias = weights_temp_.attn_k_bias;
    weights_.attn_v_proj = weights_temp_.attn_v_proj;
    weights_.attn_v_bias = weights_temp_.attn_v_bias;
    weights_.attn_o_proj = weights_temp_.attn_o_proj;
    weights_.ffn_norm = weights_temp_.ffn_norm;
    weights_.ffn_gate_proj = weights_temp_.ffn_gate_proj;
    weights_.ffn_up_proj = weights_temp_.ffn_up_proj;
    weights_.ffn_down_proj = weights_temp_.ffn_down_proj;

    built_ = true;
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_->seq_len = 0;
}

tensor_t Qwen2Model::forward_embed(const tensor_t& input_ids) {
    // input_ids: [seqlen] (int64)
    // output: [seqlen, hidden_size]

    size_t seqlen = input_ids->numel();

    // Create output tensor
    std::vector<size_t> out_shape = {seqlen, config_.hidden_size};
    tensor_t hidden_state = Tensor::create(out_shape, config_.dtype, input_ids->deviceType(), input_ids->deviceId());

    // Perform embedding lookup
    ops::embedding(hidden_state, input_ids, weights_.embed_tokens);

    return hidden_state;
}

tensor_t Qwen2Model::forward_attention(const tensor_t& hidden_state, size_t layer_idx, const std::vector<int64_t>& pos_ids) {
    size_t seqlen = hidden_state->shape()[0];
    size_t head_dim = config_.hidden_size / config_.n_head;

    // 1. RMS norm
    std::vector<size_t> hidden_shape = hidden_state->shape();
    tensor_t hidden = Tensor::create(hidden_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::rms_norm(hidden, hidden_state, weights_.attn_norm[layer_idx], config_.rms_norm_eps);

    // 2. QKV projection - reshape to [seqlen, n_head/n_kv_head, head_dim]
    // First project to [seqlen, n_head * head_dim] or [seqlen, n_kv_head * head_dim]
    std::vector<size_t> q_flat_shape = {seqlen, config_.n_head * head_dim};
    std::vector<size_t> kv_flat_shape = {seqlen, config_.n_kv_head * head_dim};
    
    tensor_t q_flat = Tensor::create(q_flat_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    tensor_t k_flat = Tensor::create(kv_flat_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    tensor_t v_flat = Tensor::create(kv_flat_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());

    ops::linear(q_flat, hidden, weights_.attn_q_proj[layer_idx], weights_.attn_q_bias[layer_idx]);
    ops::linear(k_flat, hidden, weights_.attn_k_proj[layer_idx], weights_.attn_k_bias[layer_idx]);
    ops::linear(v_flat, hidden, weights_.attn_v_proj[layer_idx], weights_.attn_v_bias[layer_idx]);

    // Reshape to [seqlen, n_head/n_kv_head, head_dim]
    std::vector<size_t> q_shape = {seqlen, config_.n_head, head_dim};
    std::vector<size_t> kv_shape = {seqlen, config_.n_kv_head, head_dim};
    
    tensor_t q = q_flat->view(q_shape);
    tensor_t k_new = k_flat->view(kv_shape);
    tensor_t v_new = v_flat->view(kv_shape);

    // 3. Apply RoPE to Q and K
    tensor_t pos_ids_tensor = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, hidden_state->deviceType(), hidden_state->deviceId());
    pos_ids_tensor->load(pos_ids.data());

    tensor_t q_rope = Tensor::create(q_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    tensor_t k_rope = Tensor::create(kv_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    
    ops::rope(q_rope, q, pos_ids_tensor, config_.rope_theta);
    ops::rope(k_rope, k_new, pos_ids_tensor, config_.rope_theta);
    
    q = q_rope;
    k_new = k_rope;

    // 4. Update KV cache
    size_t cache_start = kv_cache_->seq_len;
    size_t cache_end = cache_start + seqlen;

    // Copy k_new and v_new to cache at positions [cache_start:cache_end]
    // NOTE: k_new and v_new are [seqlen, n_kv_head, head_dim], contiguous
    //       cache is [max_seq_len, n_kv_head, head_dim], we need to copy to [cache_start:cache_end, :, :]
    size_t kv_elements = seqlen * config_.n_kv_head * head_dim;
    size_t kv_bytes = kv_elements * llaisys::utils::dsize(config_.dtype);

    // Get direct pointers to cache at the correct offset
    size_t cache_offset = cache_start * config_.n_kv_head * head_dim * llaisys::utils::dsize(config_.dtype);
    std::byte* k_cache_ptr = kv_cache_->keys[layer_idx]->data() + cache_offset;
    std::byte* v_cache_ptr = kv_cache_->values[layer_idx]->data() + cache_offset;

    std::memcpy(k_cache_ptr, k_new->data(), kv_bytes);
    std::memcpy(v_cache_ptr, v_new->data(), kv_bytes);

    // 5. Prepare full KV cache view for attention (all cached tokens + new tokens)
    tensor_t k_full = kv_cache_->keys[layer_idx]->slice(0, 0, cache_end);
    tensor_t v_full = kv_cache_->values[layer_idx]->slice(0, 0, cache_end);

    // Reshape k_full, v_full to [kvlen, n_kv_head, head_dim] where kvlen = cache_end
    std::vector<size_t> kv_full_shape = {cache_end, config_.n_kv_head, head_dim};
    tensor_t k = k_full->view(kv_full_shape);
    tensor_t v = v_full->view(kv_full_shape);

    // 6. Self-attention
    // Calculate scale
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Self-attention output: [seqlen, n_head, head_dim]
    tensor_t attn_out_3d = Tensor::create(q_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::self_attention(attn_out_3d, q, k, v, scale);

    // 7. Reshape attention output back to [seqlen, hidden_size]
    std::vector<size_t> attn_flat_shape = {seqlen, config_.hidden_size};
    tensor_t attn_out = attn_out_3d->view(attn_flat_shape);

    // 8. Output projection
    std::vector<size_t> out_shape = {seqlen, config_.hidden_size};
    tensor_t out = Tensor::create(out_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::linear(out, attn_out, weights_.attn_o_proj[layer_idx], nullptr);

    // Note: kv_cache_->seq_len is updated in infer() after all layers process the batch
    // Do NOT update it here, as each layer needs to use the same cache_start

    return out;
}

tensor_t Qwen2Model::forward_mlp(const tensor_t& hidden_state, size_t layer_idx) {
    size_t seqlen = hidden_state->shape()[0];

    // 1. RMS norm
    std::vector<size_t> hidden_shape = hidden_state->shape();
    tensor_t hidden = Tensor::create(hidden_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::rms_norm(hidden, hidden_state, weights_.ffn_norm[layer_idx], config_.rms_norm_eps);

    // 2. Project to intermediate_size
    std::vector<size_t> intermediate_shape = {seqlen, config_.intermediate_size};
    tensor_t gate = Tensor::create(intermediate_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    tensor_t up = Tensor::create(intermediate_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());

    ops::linear(gate, hidden, weights_.ffn_gate_proj[layer_idx], nullptr);
    ops::linear(up, hidden, weights_.ffn_up_proj[layer_idx], nullptr);

    // 3. SwiGLU
    tensor_t swiglu_out = Tensor::create(intermediate_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::swiglu(swiglu_out, gate, up);

    // 4. Project back to hidden_size
    std::vector<size_t> out_shape = {seqlen, config_.hidden_size};
    tensor_t mlp_out = Tensor::create(out_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::linear(mlp_out, swiglu_out, weights_.ffn_down_proj[layer_idx], nullptr);

    return mlp_out;
}

tensor_t Qwen2Model::forward_layer(const tensor_t& hidden_state, size_t layer_idx, const std::vector<int64_t>& pos_ids) {
    // Attention with residual
    tensor_t attn_out = forward_attention(hidden_state, layer_idx, pos_ids);
    
    // Add residual
    tensor_t attn_residual = Tensor::create(hidden_state->shape(), config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::add(attn_residual, hidden_state, attn_out);
    
    // MLP with residual
    tensor_t mlp_out = forward_mlp(attn_residual, layer_idx);
    
    // Add residual
    tensor_t mlp_residual = Tensor::create(attn_residual->shape(), config_.dtype, attn_residual->deviceType(), attn_residual->deviceId());
    ops::add(mlp_residual, attn_residual, mlp_out);
    
    return mlp_residual;
}

tensor_t Qwen2Model::forward_head(const tensor_t& hidden_state) {
    size_t seqlen = hidden_state->shape()[0];
    std::vector<size_t> out_shape = {seqlen, config_.vocab_size};
    tensor_t logits = Tensor::create(out_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());

    // RMS norm
    std::vector<size_t> hidden_shape = hidden_state->shape();
    tensor_t hidden = Tensor::create(hidden_shape, config_.dtype, hidden_state->deviceType(), hidden_state->deviceId());
    ops::rms_norm(hidden, hidden_state, weights_.norm_f, config_.rms_norm_eps);

    // Linear projection to vocab
    ops::linear(logits, hidden, weights_.lm_head, nullptr);

    return logits;
}

int64_t Qwen2Model::sample(const tensor_t& logits, int top_k, float top_p, float temperature) {
    // Get last token logits [vocab_size]
    size_t seqlen = logits->shape()[0];
    tensor_t last_logits = logits->slice(0, seqlen - 1, seqlen);
    
    // Flatten to 1D [vocab_size]
    std::vector<size_t> flat_shape = {config_.vocab_size};
    tensor_t flat_logits = last_logits->view(flat_shape);

    // For sampling, we need to work on CPU
    // Copy logits to CPU if needed
    tensor_t cpu_logits = flat_logits;
    if (flat_logits->deviceType() != LLAISYS_DEVICE_CPU) {
        cpu_logits = Tensor::create(flat_shape, config_.dtype, LLAISYS_DEVICE_CPU, 0);
        // Copy from device to CPU using runtime API
        size_t bytes = config_.vocab_size * llaisys::utils::dsize(config_.dtype);
        llaisys::core::context().runtime().api()->memcpy_sync(
            cpu_logits->data(), flat_logits->data(), bytes, LLAISYS_MEMCPY_D2H);
    }

    // Read logits into float array for processing
    std::vector<float> logits_f32(config_.vocab_size);
    size_t elem_size = llaisys::utils::dsize(config_.dtype);
    
    for (size_t i = 0; i < config_.vocab_size; i++) {
        const std::byte* ptr = cpu_logits->data() + i * elem_size;
        switch (config_.dtype) {
        case LLAISYS_DTYPE_F32:
            logits_f32[i] = *reinterpret_cast<const float*>(ptr);
            break;
        case LLAISYS_DTYPE_F16: {
            llaisys::fp16_t val = *reinterpret_cast<const llaisys::fp16_t*>(ptr);
            logits_f32[i] = llaisys::utils::cast<float>(val);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            llaisys::bf16_t val = *reinterpret_cast<const llaisys::bf16_t*>(ptr);
            logits_f32[i] = llaisys::utils::cast<float>(val);
            break;
        }
        default:
            // Fallback: try to read as float
            logits_f32[i] = *reinterpret_cast<const float*>(ptr);
            break;
        }
    }

    // 1. Apply temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (size_t i = 0; i < config_.vocab_size; i++) {
            logits_f32[i] *= inv_temp;
        }
    }

    // 2. Find max for numerical stability
    float max_logit = logits_f32[0];
    for (size_t i = 1; i < config_.vocab_size; i++) {
        if (logits_f32[i] > max_logit) {
            max_logit = logits_f32[i];
        }
    }

    // 3. Compute softmax (exp and sum)
    std::vector<float> probs(config_.vocab_size);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < config_.vocab_size; i++) {
        float exp_val = std::exp(logits_f32[i] - max_logit);
        probs[i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize
    for (size_t i = 0; i < config_.vocab_size; i++) {
        probs[i] /= sum_exp;
    }

    // 4. Apply top_k filtering
    // Special case: if top_k == 1, we can directly return argmax without sampling
    if (top_k == 1) {
        int64_t max_idx = 0;
        float max_prob = probs[0];
        for (size_t i = 1; i < config_.vocab_size; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_idx = static_cast<int64_t>(i);
            }
        }
        return max_idx;
    }
    
    if (top_k > 1 && top_k < static_cast<int>(config_.vocab_size)) {
        // Create index-probability pairs and sort by probability (descending)
        std::vector<std::pair<size_t, float>> idx_probs;
        for (size_t i = 0; i < config_.vocab_size; i++) {
            idx_probs.push_back({i, probs[i]});
        }
        
        // Partial sort to get top_k (nth_element + sort first k elements)
        std::nth_element(idx_probs.begin(), idx_probs.begin() + top_k, idx_probs.end(),
                        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                            return a.second > b.second;
                        });
        // Sort the top_k elements for consistency
        std::sort(idx_probs.begin(), idx_probs.begin() + top_k,
                 [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                     return a.second > b.second;
                 });
        
        // Zero out probabilities not in top_k
        std::vector<bool> keep(config_.vocab_size, false);
        float top_k_sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            keep[idx_probs[i].first] = true;
            top_k_sum += idx_probs[i].second;
        }
        
        // Renormalize - ensure top_k_sum is not zero
        if (top_k_sum > 0.0f) {
            for (size_t i = 0; i < config_.vocab_size; i++) {
                if (keep[i]) {
                    probs[i] /= top_k_sum;
                } else {
                    probs[i] = 0.0f;
                }
            }
        }
    }

    // 5. Apply top_p (nucleus) filtering
    if (top_p < 1.0f) {
        // Create index-probability pairs and sort by probability (descending)
        std::vector<std::pair<size_t, float>> idx_probs;
        for (size_t i = 0; i < config_.vocab_size; i++) {
            if (probs[i] > 0.0f) {
                idx_probs.push_back({i, probs[i]});
            }
        }
        
        std::sort(idx_probs.begin(), idx_probs.end(),
                 [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                     return a.second > b.second;
                 });
        
        // Find cumulative sum threshold
        float cumsum = 0.0f;
        size_t cutoff = idx_probs.size();
        for (size_t i = 0; i < idx_probs.size(); i++) {
            cumsum += idx_probs[i].second;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        std::vector<bool> keep(config_.vocab_size, false);
        float top_p_sum = 0.0f;
        for (size_t i = 0; i < cutoff; i++) {
            keep[idx_probs[i].first] = true;
            top_p_sum += idx_probs[i].second;
        }
        
        // Renormalize
        for (size_t i = 0; i < config_.vocab_size; i++) {
            if (keep[i]) {
                probs[i] /= top_p_sum;
            } else {
                probs[i] = 0.0f;
            }
        }
    }

    // 6. Sample from the probability distribution
    // Count valid tokens and calculate total probability
    std::vector<size_t> valid_indices;
    float total_prob = 0.0f;
    for (size_t i = 0; i < config_.vocab_size; i++) {
        if (probs[i] > 0.0f) {
            valid_indices.push_back(i);
            total_prob += probs[i];
        }
    }
    
    // If no valid probabilities, return 0 (should not happen, but handle gracefully)
    if (valid_indices.empty() || total_prob <= 0.0f) {
        return 0;
    }
    
    // If only one valid token, return it directly (no need for random sampling)
    if (valid_indices.size() == 1) {
        return static_cast<int64_t>(valid_indices[0]);
    }
    
    // If temperature is very small or zero, use argmax instead
    if (temperature <= 0.0f) {
        // Find argmax among valid tokens
        int64_t max_idx = static_cast<int64_t>(valid_indices[0]);
        float max_prob = probs[valid_indices[0]];
        for (size_t idx : valid_indices) {
            if (probs[idx] > max_prob) {
                max_prob = probs[idx];
                max_idx = static_cast<int64_t>(idx);
            }
        }
        return max_idx;
    }
    
    // Random sampling: use multinomial sampling
    // Generate random value in [0, total_prob)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, total_prob);
    
    float r = dis(gen);
    float cumsum = 0.0f;
    int64_t token_id = static_cast<int64_t>(valid_indices.back());  // Fallback to last valid token
    
    // Find the token corresponding to the random value
    // Traverse through valid tokens and find the one where cumsum >= r
    for (size_t idx : valid_indices) {
        cumsum += probs[idx];
        if (r < cumsum) {  // Use < instead of <= to avoid edge case issues
            token_id = static_cast<int64_t>(idx);
            break;
        }
    }
    
    return token_id;
}

int64_t Qwen2Model::infer(const std::vector<int64_t>& input_ids, int top_k, float top_p, float temperature) {
    size_t seqlen = input_ids.size();

    // Create position IDs
    std::vector<int64_t> pos_ids(seqlen);
    for (size_t i = 0; i < seqlen; i++) {
        pos_ids[i] = kv_cache_->seq_len + i;
    }

    // Embedding
    tensor_t input_ids_tensor = Tensor::create({seqlen}, LLAISYS_DTYPE_I64,
                                                kv_cache_->keys[0]->deviceType(), kv_cache_->keys[0]->deviceId());
    input_ids_tensor->load(input_ids.data());

    tensor_t hidden_state = forward_embed(input_ids_tensor);

    // Forward through all layers
    for (size_t layer = 0; layer < config_.n_layer; layer++) {
        hidden_state = forward_layer(hidden_state, layer, pos_ids);
    }

    // Update KV cache length AFTER all layers have processed the batch
    // This ensures all layers use the same cache_start position
    kv_cache_->seq_len += seqlen;

    // Head
    tensor_t logits = forward_head(hidden_state);

    // Sample
    return sample(logits, top_k, top_p, temperature);
}

} // namespace qwen2
} // namespace models
} // namespace llaisys
