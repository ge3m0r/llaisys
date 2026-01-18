#include "llaisys_models.hpp"

using namespace llaisys::models::qwen2;

// Helper to convert C struct to C++ config
static Qwen2Config convert_meta(const LlaisysQwen2Meta* meta) {
    Qwen2Config config;
    config.n_layer = meta->nlayer;
    config.hidden_size = meta->hs;
    config.n_head = meta->nh;
    config.n_kv_head = meta->nkvh;
    config.intermediate_size = meta->di;
    config.max_seq_len = meta->maxseq;
    config.vocab_size = meta->voc;
    config.rms_norm_eps = meta->epsilon;
    config.rope_theta = meta->theta;
    config.dtype = meta->dtype;
    config.eos_token = meta->end_token;
    return config;
}

__C {
    __export LlaisysQwen2Model *llaisysQwen2ModelCreate(
            const LlaisysQwen2Meta *meta,
            llaisysDeviceType_t device,
            int *device_ids,
            int ndevice) {

        Qwen2Config config = convert_meta(meta);
        model_t model = std::make_shared<Qwen2Model>(config, device, device_ids ? device_ids[0] : 0);

        auto wrapper = new LlaisysQwen2Model{model};
        return wrapper;
    }

    __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        auto wrapper = new LlaisysQwen2Weights{model};
        return wrapper;
    }

    __export void llaisysQwen2WeightsDestroy(LlaisysQwen2Weights *weights) {
        delete weights;
    }

    __export void llaisysQwen2SetEmbedTokens(
            LlaisysQwen2Model *model,
            llaisysTensor_t tensor) {
        model->model->set_weights_field("embed_tokens", tensor->tensor);
    }

    __export void llaisysQwen2SetOutEmbed(
            LlaisysQwen2Model *model,
            llaisysTensor_t tensor) {
        model->model->set_weights_field("lm_head", tensor->tensor);
    }

    __export void llaisysQwen2SetOutNormW(
            LlaisysQwen2Model *model,
            llaisysTensor_t tensor) {
        model->model->set_weights_field("norm_f", tensor->tensor);
    }

    __export void llaisysQwen2SetAttnNormW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_norm(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnQW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_q_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnQB(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_q_bias(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnKW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_k_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnKB(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_k_bias(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnVW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_v_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnVB(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_v_bias(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetAttnOW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_attn_o_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetMlpNormW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_ffn_norm(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetMlpGateW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_ffn_gate_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetMlpUpW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_ffn_up_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2SetMlpDownW(
            LlaisysQwen2Model *model,
            llaisysTensor_t *tensors,
            size_t nlayer) {
        for (size_t i = 0; i < nlayer; i++) {
            model->model->set_ffn_down_proj(i, tensors[i]->tensor);
        }
    }

    __export void llaisysQwen2ModelBuild(LlaisysQwen2Model *model) {
        model->model->build();
    }

    __export int64_t llaisysQwen2ModelInfer(
            LlaisysQwen2Model *model,
            int64_t *token_ids,
            size_t ntoken,
            int top_k,
            float top_p,
            float temperature) {

        std::vector<int64_t> input_ids(token_ids, token_ids + ntoken);
        int64_t next_token = model->model->infer(input_ids, top_k, top_p, temperature);
        return next_token;
    }

    __export void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        model->model->reset_kv_cache();
    }
}
