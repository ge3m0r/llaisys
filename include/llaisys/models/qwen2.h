#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    typedef struct LlaisysQwen2Model LlaisysQwen2Model;

    typedef struct LlaisysQwen2Weights {
        LlaisysQwen2Model *model;
    } LlaisysQwen2Weights;

    typedef struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    } LlaisysQwen2Meta;

    __export LlaisysQwen2Model *llaisysQwen2ModelCreate(
            const LlaisysQwen2Meta *meta,
            llaisysDeviceType_t device,
            int *device_ids,
            int ndevice);

    __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model);

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model);

    __export void llaisysQwen2WeightsDestroy(LlaisysQwen2Weights *weights);

    // Weight setters - single tensor
    __export void llaisysQwen2SetEmbedTokens(LlaisysQwen2Model *model, llaisysTensor_t tensor);
    __export void llaisysQwen2SetOutEmbed(LlaisysQwen2Model *model, llaisysTensor_t tensor);
    __export void llaisysQwen2SetOutNormW(LlaisysQwen2Model *model, llaisysTensor_t tensor);

    // Weight setters - arrays
    __export void llaisysQwen2SetAttnNormW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnQW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnQB(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnKW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnKB(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnVW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnVB(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetAttnOW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);

    __export void llaisysQwen2SetMlpNormW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetMlpGateW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetMlpUpW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);
    __export void llaisysQwen2SetMlpDownW(LlaisysQwen2Model *model, llaisysTensor_t *tensors, size_t nlayer);

    __export void llaisysQwen2ModelBuild(LlaisysQwen2Model *model);

    __export int64_t llaisysQwen2ModelInfer(
            LlaisysQwen2Model *model,
            int64_t *token_ids,
            size_t ntoken,
            int top_k,
            float top_p,
            float temperature);

    __export void llaisysQwen2ModelReset(LlaisysQwen2Model *model);
}
#endif // LLAISYS_MODELS_QWEN2_H
