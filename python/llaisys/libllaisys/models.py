from ctypes import c_int, c_int64, c_size_t, c_float, POINTER, Structure
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


# Define pointer types
LlaisysQwen2Meta_p = POINTER(LlaisysQwen2Meta)


class LlaisysQwen2Model(Structure):
    pass


LlaisysQwen2Model_p = POINTER(LlaisysQwen2Model)


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("model", LlaisysQwen2Model_p),
    ]


LlaisysQwen2Weights_p = POINTER(LlaisysQwen2Weights)


def load_models(lib):
    # Model creation and destruction
    lib.llaisysQwen2ModelCreate.argtypes = [
        LlaisysQwen2Meta_p,  # meta
        c_int,  # device
        POINTER(c_int),  # device_ids
        c_int,  # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights_p

    lib.llaisysQwen2WeightsDestroy.argtypes = [LlaisysQwen2Weights_p]
    lib.llaisysQwen2WeightsDestroy.restype = None

    # Weight setters - single tensor
    lib.llaisysQwen2SetEmbedTokens.argtypes = [LlaisysQwen2Model_p, llaisysTensor_t]
    lib.llaisysQwen2SetEmbedTokens.restype = None

    lib.llaisysQwen2SetOutEmbed.argtypes = [LlaisysQwen2Model_p, llaisysTensor_t]
    lib.llaisysQwen2SetOutEmbed.restype = None

    lib.llaisysQwen2SetOutNormW.argtypes = [LlaisysQwen2Model_p, llaisysTensor_t]
    lib.llaisysQwen2SetOutNormW.restype = None

    # Weight setters - arrays
    lib.llaisysQwen2SetAttnNormW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnNormW.restype = None

    lib.llaisysQwen2SetAttnQW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnQW.restype = None

    lib.llaisysQwen2SetAttnQB.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnQB.restype = None

    lib.llaisysQwen2SetAttnKW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnKW.restype = None

    lib.llaisysQwen2SetAttnKB.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnKB.restype = None

    lib.llaisysQwen2SetAttnVW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnVW.restype = None

    lib.llaisysQwen2SetAttnVB.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnVB.restype = None

    lib.llaisysQwen2SetAttnOW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetAttnOW.restype = None

    lib.llaisysQwen2SetMlpNormW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetMlpNormW.restype = None

    lib.llaisysQwen2SetMlpGateW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetMlpGateW.restype = None

    lib.llaisysQwen2SetMlpUpW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetMlpUpW.restype = None

    lib.llaisysQwen2SetMlpDownW.argtypes = [LlaisysQwen2Model_p, POINTER(llaisysTensor_t), c_size_t]
    lib.llaisysQwen2SetMlpDownW.restype = None

    # Model operations
    lib.llaisysQwen2ModelBuild.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelBuild.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model_p, POINTER(c_int64), c_size_t, c_int, c_float, c_float]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelReset.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelReset.restype = None
