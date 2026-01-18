from typing import Sequence, List, Dict, Any
from ctypes import POINTER, cast, byref, c_void_p, c_longlong
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import (
    LlaisysQwen2Meta, LlaisysQwen2Model_p,
    load_models
)
from ..libllaisys.tensor import llaisysTensor_t, create_tensor, Tensor
from pathlib import Path
import safetensors
import numpy as np


class Qwen2:

    # Qwen2 1.5B configuration
    CONFIG = {
        "n_layer": 28,
        "hidden_size": 1536,
        "n_head": 12,
        "n_kv_head": 2,
        "intermediate_size": 8960,
        "max_seq_len": 131072,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "eos_token": 151643,
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # Load model ctypes functions
        load_models(LIB_LLAISYS)

        model_path = Path(model_path)
        dtype = DataType.BF16
        
        # Create model metadata
        meta = LlaisysQwen2Meta()
        meta.dtype = dtype.value
        meta.nlayer = self.CONFIG["n_layer"]
        meta.hs = self.CONFIG["hidden_size"]
        meta.nh = self.CONFIG["n_head"]
        meta.nkvh = self.CONFIG["n_kv_head"]
        meta.dh = self.CONFIG["hidden_size"] // self.CONFIG["n_head"]
        meta.di = self.CONFIG["intermediate_size"]
        meta.maxseq = self.CONFIG["max_seq_len"]
        meta.voc = self.CONFIG["vocab_size"]
        meta.epsilon = self.CONFIG["rms_norm_eps"]
        meta.theta = self.CONFIG["rope_theta"]
        meta.end_token = self.CONFIG["eos_token"]
        
        # Create model
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta,
            device.value,
            None,
            1
        )

        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")
            
        # Load model weights
        self._all_tensors = []
        self._load_weights(model_path, dtype, device)

        # Build model after loading weights
        LIB_LLAISYS.llaisysQwen2ModelBuild(self.model)

        self.dtype = dtype
        self.device = device

    def __del__(self):
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None

    def _create_and_load_tensor(self, name: str, shape: List[int], weights_dict: Dict, dtype: DataType, device: DeviceType) -> llaisysTensor_t:
        if name not in weights_dict:
            return None

        data = weights_dict[name]

        # Convert to float32 for loading
        dtype_str = str(data.dtype)
        if 'bfloat16' in dtype_str or 'bf16' in dtype_str.lower():
            data = data.astype(np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Convert float32 to BF16 bytes if tensor dtype is BF16
        if dtype == DataType.BF16:
            import struct
            # Convert float32 to BF16 with proper rounding
            # Pre-allocate bytearray to avoid memory issues with large tensors
            num_elements = data.size
            bf16_bytes = bytearray(num_elements * 2)  # 2 bytes per BF16 value
            
            # Use numpy for efficient conversion
            # Convert to uint32 to extract bits
            f32_view = data.view(np.uint32)
            
            for i, f32_bits in enumerate(f32_view.flat):
                # Extract sign, exponent, mantissa
                sign = (f32_bits >> 31) & 0x1
                exponent = (f32_bits >> 23) & 0xFF
                mantissa = f32_bits & 0x7FFFFF

                if exponent == 0:
                    # Zero or subnormal
                    bf16_bits = sign << 15
                elif exponent == 255:
                    # Inf or NaN - preserve sign and exponent bits, truncate mantissa
                    bf16_bits = (sign << 15) | 0x7F << 7 | (mantissa >> 16)
                else:
                    # Normal number
                    # Round to nearest even: add 0.5 ULP (unit in last place)
                    # The 16th bit of mantissa (bit 15 of the lower 16 bits) is the rounding bit
                    rounding_bit = (mantissa >> 15) & 1
                    rounded_mantissa = mantissa + rounding_bit

                    # Handle overflow from rounding
                    if rounded_mantissa & 0x800000:  # Carry into exponent
                        rounded_mantissa = 0
                        exponent += 1
                        if exponent == 255:  # Overflow to infinity
                            exponent = 254
                            rounded_mantissa = 0x7FFFFF

                    # BF16: sign (1 bit) + exponent (8 bits, bias 127) + mantissa (7 bits)
                    bf16_bits = (sign << 15) | (exponent << 7) | (rounded_mantissa >> 16)

                # Pack directly into pre-allocated bytearray
                struct.pack_into('<H', bf16_bytes, i * 2, bf16_bits & 0xFFFF)
            
            data_bytes = bytes(bf16_bytes)
        else:
            data_bytes = data.tobytes()

        tensor = create_tensor(shape, dtype, device)
        if tensor is None or tensor.ptr is None:
            raise RuntimeError(f"Failed to create tensor for {name}")

        tensor.load(data_bytes)
        ptr = tensor.ptr
        
        if ptr is None or (isinstance(ptr, int) and ptr == 0):
            raise RuntimeError(f"Invalid tensor pointer for {name}")
        
        self._all_tensors.append(tensor)
        return ptr

    def _load_weights(self, model_path: Path, dtype: DataType, device: DeviceType):
        weights_dict = {}

        def bf16_bytes_to_float32(bf16_bytes):
            import struct
            count = len(bf16_bytes) // 2
            float32_array = np.zeros(count, dtype=np.float32)
            for i in range(count):
                bf16_bits = struct.unpack("<H", bf16_bytes[2*i:2*i+2])[0]
                f32_bits = bf16_bits << 16
                float32_array[i] = struct.unpack("<f", struct.pack("<I", f32_bits))[0]
            return float32_array

        for file in sorted(model_path.glob("*.safetensors")):
            with open(file, "rb") as f:
                import struct, json
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
                data_block_offset = 8 + header_size
                
                for name_, tensor_info in header.items():
                    if name_ == "__metadata__":
                        continue
                    if not isinstance(tensor_info, dict) or "dtype" not in tensor_info:
                        continue
                    
                    dtype_str = tensor_info["dtype"]
                    shape = tensor_info["shape"]
                    begin, end = tensor_info["data_offsets"]
                    
                    f.seek(data_block_offset + begin)
                    raw_bytes = f.read(end - begin)
                    
                    tensor_np = None
                    if dtype_str == "BF16":
                        tensor_np = bf16_bytes_to_float32(raw_bytes).reshape(shape)
                    elif dtype_str == "F32":
                        tensor_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
                    else:
                        print(f"Warning: Unsupported dtype {dtype_str} for tensor {name_}")

                    if tensor_np is not None:
                        weights_dict[name_] = tensor_np

        # Load embedding weights
        embed_names = ["model.embed_tokens.weight", "embed_tokens.weight"]
        embed_tokens = None
        for name in embed_names:
            embed_tokens = self._create_and_load_tensor(name, [self.CONFIG["vocab_size"], self.CONFIG["hidden_size"]], weights_dict, dtype, device)
            if embed_tokens:
                break
        if embed_tokens:
            LIB_LLAISYS.llaisysQwen2SetEmbedTokens(self.model, embed_tokens)

        # Load output weights (lm_head)
        lm_head_names = ["lm_head.weight", "model.lm_head.weight"]
        lm_head = None
        for name in lm_head_names:
            lm_head = self._create_and_load_tensor(name, [self.CONFIG["vocab_size"], self.CONFIG["hidden_size"]], weights_dict, dtype, device)
            if lm_head:
                break
        if lm_head:
            LIB_LLAISYS.llaisysQwen2SetOutEmbed(self.model, lm_head)
        elif embed_tokens: # Tied weights
            LIB_LLAISYS.llaisysQwen2SetOutEmbed(self.model, embed_tokens)
        else:
            raise ValueError("Cannot find output embedding weights (lm_head or tied).")

        # Load final norm
        norm_f = self._create_and_load_tensor("model.norm.weight", [self.CONFIG["hidden_size"]], weights_dict, dtype, device)
        if norm_f:
            LIB_LLAISYS.llaisysQwen2SetOutNormW(self.model, norm_f)

        # Load layer weights
        nlayer = self.CONFIG["n_layer"]
        
        def to_ctypes_array(tensor_list):
            if not tensor_list: return None
            if any(t is None for t in tensor_list):
                # This is a simplified check. For optional weights like biases, this needs refinement.
                # For now, we assume all weights in a list are required if the list is passed.
                # A more robust solution would be to not add None to the list in the first place
                # and check if the list is empty before calling the C function.
                pass
            
            # Filter out None values for optional weights like biases
            filtered_list = [t for t in tensor_list if t is not None]
            if not filtered_list:
                return None

            ArrayType = (llaisysTensor_t * len(filtered_list))
            return ArrayType(*filtered_list)

        # Pre-collect all layer tensors
        layer_tensors = {
            "attn_norm": [], "attn_q_w": [], "attn_q_b": [], "attn_k_w": [], "attn_k_b": [],
            "attn_v_w": [], "attn_v_b": [], "attn_o_w": [], "mlp_norm": [], "mlp_gate_w": [],
            "mlp_up_w": [], "mlp_down_w": []
        }

        for i in range(nlayer):
            prefix = f"model.layers.{i}."
            
            # Attention
            layer_tensors["attn_norm"].append(self._create_and_load_tensor(f"{prefix}input_layernorm.weight", [self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["attn_q_w"].append(self._create_and_load_tensor(f"{prefix}self_attn.q_proj.weight", [self.CONFIG["n_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"]), self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["attn_q_b"].append(self._create_and_load_tensor(f"{prefix}self_attn.q_proj.bias", [self.CONFIG["n_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"])], weights_dict, dtype, device))
            layer_tensors["attn_k_w"].append(self._create_and_load_tensor(f"{prefix}self_attn.k_proj.weight", [self.CONFIG["n_kv_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"]), self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["attn_k_b"].append(self._create_and_load_tensor(f"{prefix}self_attn.k_proj.bias", [self.CONFIG["n_kv_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"])], weights_dict, dtype, device))
            layer_tensors["attn_v_w"].append(self._create_and_load_tensor(f"{prefix}self_attn.v_proj.weight", [self.CONFIG["n_kv_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"]), self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["attn_v_b"].append(self._create_and_load_tensor(f"{prefix}self_attn.v_proj.bias", [self.CONFIG["n_kv_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"])], weights_dict, dtype, device))
            layer_tensors["attn_o_w"].append(self._create_and_load_tensor(f"{prefix}self_attn.o_proj.weight", [self.CONFIG["hidden_size"], self.CONFIG["n_head"] * (self.CONFIG["hidden_size"] // self.CONFIG["n_head"])], weights_dict, dtype, device))

            # MLP
            layer_tensors["mlp_norm"].append(self._create_and_load_tensor(f"{prefix}post_attention_layernorm.weight", [self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["mlp_gate_w"].append(self._create_and_load_tensor(f"{prefix}mlp.gate_proj.weight", [self.CONFIG["intermediate_size"], self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["mlp_up_w"].append(self._create_and_load_tensor(f"{prefix}mlp.up_proj.weight", [self.CONFIG["intermediate_size"], self.CONFIG["hidden_size"]], weights_dict, dtype, device))
            layer_tensors["mlp_down_w"].append(self._create_and_load_tensor(f"{prefix}mlp.down_proj.weight", [self.CONFIG["hidden_size"], self.CONFIG["intermediate_size"]], weights_dict, dtype, device))

        # Set weights in C++
        LIB_LLAISYS.llaisysQwen2SetAttnNormW(self.model, to_ctypes_array(layer_tensors["attn_norm"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetAttnQW(self.model, to_ctypes_array(layer_tensors["attn_q_w"]), nlayer)
        if any(layer_tensors["attn_q_b"]): LIB_LLAISYS.llaisysQwen2SetAttnQB(self.model, to_ctypes_array(layer_tensors["attn_q_b"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetAttnKW(self.model, to_ctypes_array(layer_tensors["attn_k_w"]), nlayer)
        if any(layer_tensors["attn_k_b"]): LIB_LLAISYS.llaisysQwen2SetAttnKB(self.model, to_ctypes_array(layer_tensors["attn_k_b"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetAttnVW(self.model, to_ctypes_array(layer_tensors["attn_v_w"]), nlayer)
        if any(layer_tensors["attn_v_b"]): LIB_LLAISYS.llaisysQwen2SetAttnVB(self.model, to_ctypes_array(layer_tensors["attn_v_b"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetAttnOW(self.model, to_ctypes_array(layer_tensors["attn_o_w"]), nlayer)
        
        LIB_LLAISYS.llaisysQwen2SetMlpNormW(self.model, to_ctypes_array(layer_tensors["mlp_norm"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetMlpGateW(self.model, to_ctypes_array(layer_tensors["mlp_gate_w"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetMlpUpW(self.model, to_ctypes_array(layer_tensors["mlp_up_w"]), nlayer)
        LIB_LLAISYS.llaisysQwen2SetMlpDownW(self.model, to_ctypes_array(layer_tensors["mlp_down_w"]), nlayer)

    def generate(self, token_ids: Sequence[int], max_new_tokens: int = 128, top_p: float = 0.8, top_k: int = 50, temperature: float = 0.8) -> List[int]:
        if not self.model:
            raise RuntimeError("Model is not initialized.")

        # Reset model state for new generation
        LIB_LLAISYS.llaisysQwen2ModelReset(self.model)

        # Process all prompt tokens at once (batch processing)
        input_arr = (c_longlong * len(token_ids))(*token_ids)
        first_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model, input_arr, len(token_ids), top_k, top_p, temperature)

        outputs = list(token_ids)  # Include input tokens in output
        next_token_id = first_token

        for _ in range(max_new_tokens):
            outputs.append(next_token_id)

            if next_token_id == self.CONFIG["eos_token"]:
                break

            # Generate next token (single token at a time for generation)
            input_arr = (c_longlong * 1)(next_token_id)
            next_token_id = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, input_arr, 1, top_k, top_p, temperature)

        return outputs
