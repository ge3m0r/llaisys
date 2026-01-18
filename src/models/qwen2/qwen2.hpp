#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"

#include <vector>
#include <memory>

namespace llaisys {
namespace models {
namespace qwen2 {

struct Qwen2Config {
    size_t n_layer;
    size_t hidden_size;
    size_t n_head;
    size_t n_kv_head;
    size_t intermediate_size;
    size_t max_seq_len;
    size_t vocab_size;
    float rms_norm_eps;
    float rope_theta;
    llaisysDataType_t dtype;
    int64_t eos_token;
};

using tensor_t = std::shared_ptr<Tensor>;
using model_t = std::shared_ptr<class Qwen2Model>;

class Qwen2Model {
public:
    Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    // Weight setters
    void set_weights_field(const std::string& name, const tensor_t& tensor);
    void set_attn_norm(size_t layer, const tensor_t& tensor);
    void set_attn_q_proj(size_t layer, const tensor_t& tensor);
    void set_attn_q_bias(size_t layer, const tensor_t& tensor);
    void set_attn_k_proj(size_t layer, const tensor_t& tensor);
    void set_attn_k_bias(size_t layer, const tensor_t& tensor);
    void set_attn_v_proj(size_t layer, const tensor_t& tensor);
    void set_attn_v_bias(size_t layer, const tensor_t& tensor);
    void set_attn_o_proj(size_t layer, const tensor_t& tensor);
    void set_ffn_norm(size_t layer, const tensor_t& tensor);
    void set_ffn_gate_proj(size_t layer, const tensor_t& tensor);
    void set_ffn_up_proj(size_t layer, const tensor_t& tensor);
    void set_ffn_down_proj(size_t layer, const tensor_t& tensor);

    void build();
    int64_t infer(const std::vector<int64_t>& input_ids, int top_k = 0, float top_p = 1.0f, float temperature = 1.0f);
    void reset_kv_cache();

private:
    Qwen2Config config_;
    
    // Temporary weight storage (before build)
    struct WeightsTemp {
        tensor_t embed_tokens;
        tensor_t lm_head;
        tensor_t norm_f;
        std::vector<tensor_t> attn_norm;
        std::vector<tensor_t> attn_q_proj;
        std::vector<tensor_t> attn_q_bias;
        std::vector<tensor_t> attn_k_proj;
        std::vector<tensor_t> attn_k_bias;
        std::vector<tensor_t> attn_v_proj;
        std::vector<tensor_t> attn_v_bias;
        std::vector<tensor_t> attn_o_proj;
        std::vector<tensor_t> ffn_norm;
        std::vector<tensor_t> ffn_gate_proj;
        std::vector<tensor_t> ffn_up_proj;
        std::vector<tensor_t> ffn_down_proj;
    } weights_temp_;

    // Built weights (after build)
    struct Weights {
        tensor_t embed_tokens;
        tensor_t lm_head;
        tensor_t norm_f;
        std::vector<tensor_t> attn_norm;
        std::vector<tensor_t> attn_q_proj;
        std::vector<tensor_t> attn_q_bias;
        std::vector<tensor_t> attn_k_proj;
        std::vector<tensor_t> attn_k_bias;
        std::vector<tensor_t> attn_v_proj;
        std::vector<tensor_t> attn_v_bias;
        std::vector<tensor_t> attn_o_proj;
        std::vector<tensor_t> ffn_norm;
        std::vector<tensor_t> ffn_gate_proj;
        std::vector<tensor_t> ffn_up_proj;
        std::vector<tensor_t> ffn_down_proj;
    } weights_;

    // KV Cache
    struct KVCache {
        size_t seq_len;
        std::vector<tensor_t> keys;
        std::vector<tensor_t> values;
    };
    std::shared_ptr<KVCache> kv_cache_;

    bool built_ = false;

    // Internal forward methods
    tensor_t forward_embed(const tensor_t& input_ids);
    tensor_t forward_layer(const tensor_t& hidden_state, size_t layer_idx, const std::vector<int64_t>& pos_ids);
    tensor_t forward_attention(const tensor_t& hidden_state, size_t layer_idx, const std::vector<int64_t>& pos_ids);
    tensor_t forward_mlp(const tensor_t& hidden_state, size_t layer_idx);
    tensor_t forward_head(const tensor_t& hidden_state);
    int64_t sample(const tensor_t& logits, int top_k = 0, float top_p = 1.0f, float temperature = 1.0f);
};

} // namespace qwen2
} // namespace models
} // namespace llaisys
