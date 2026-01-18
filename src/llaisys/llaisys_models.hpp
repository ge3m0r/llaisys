#pragma once
#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../models/qwen2/qwen2.hpp"

__C {
    struct LlaisysQwen2Model {
        std::shared_ptr<llaisys::models::qwen2::Qwen2Model> model;
    };
}
