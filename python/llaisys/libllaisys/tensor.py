from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Handle type
llaisysTensor_t = c_void_p


class Tensor:
    """Helper class for creating and managing tensors"""
    def __init__(self, tensor_ptr):
        self._tensor = tensor_ptr

    @property
    def ptr(self):
        """Get the raw tensor pointer for C API calls"""
        return self._tensor

    def load(self, data):
        """Load data from bytes into tensor"""
        # Lazy import to avoid circular dependency
        from . import LIB_LLAISYS
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    def __del__(self):
        try:
            from . import LIB_LLAISYS
            if hasattr(self, '_tensor') and self._tensor:
                LIB_LLAISYS.tensorDestroy(self._tensor)
        except:
            pass


def load_tensor(lib):
    lib.tensorCreate.argtypes = [
        POINTER(c_size_t),  # shape
        c_size_t,  # ndim
        llaisysDataType_t,  # dtype
        llaisysDeviceType_t,  # device_type
        c_int,  # device_id
    ]
    lib.tensorCreate.restype = llaisysTensor_t

    # Function: tensorDestroy
    lib.tensorDestroy.argtypes = [llaisysTensor_t]
    lib.tensorDestroy.restype = None

    # Function: tensorGetData
    lib.tensorGetData.argtypes = [llaisysTensor_t]
    lib.tensorGetData.restype = c_void_p

    # Function: tensorGetNdim
    lib.tensorGetNdim.argtypes = [llaisysTensor_t]
    lib.tensorGetNdim.restype = c_size_t

    # Function: tensorGetShape
    lib.tensorGetShape.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorGetShape.restype = None

    # Function: tensorGetStrides
    lib.tensorGetStrides.argtypes = [llaisysTensor_t, POINTER(c_ssize_t)]
    lib.tensorGetStrides.restype = None

    # Function: tensorGetDataType
    lib.tensorGetDataType.argtypes = [llaisysTensor_t]
    lib.tensorGetDataType.restype = llaisysDataType_t

    # Function: tensorGetDeviceType
    lib.tensorGetDeviceType.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceType.restype = llaisysDeviceType_t

    # Function: tensorGetDeviceId
    lib.tensorGetDeviceId.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceId.restype = c_int

    # Function: tensorDebug
    lib.tensorDebug.argtypes = [llaisysTensor_t]
    lib.tensorDebug.restype = None

    # Function: tensorIsContiguous
    lib.tensorIsContiguous.argtypes = [llaisysTensor_t]
    lib.tensorIsContiguous.restype = c_uint8

    # Function: tensorLoad
    lib.tensorLoad.argtypes = [llaisysTensor_t, c_void_p]
    lib.tensorLoad.restype = None

    # Function: tensorView(llaisysTensor_t tensor, size_t *shape);
    lib.tensorView.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorView.restype = llaisysTensor_t

    # Function: tensorPermute(llaisysTensor_t tensor, size_t *order);
    lib.tensorPermute.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorPermute.restype = llaisysTensor_t

    # Function: tensorSlice(llaisysTensor_t tensor,
    #                     size_t dim, size_t start, size_t end);
    lib.tensorSlice.argtypes = [
        llaisysTensor_t,  # tensor handle
        c_size_t,  # dim  : which axis to slice
        c_size_t,  # start: inclusive
        c_size_t,  # end  : exclusive
    ]
    lib.tensorSlice.restype = llaisysTensor_t


def create_tensor(shape, dtype, device):
    """Helper function to create a tensor from Python"""
    from . import LIB_LLAISYS

    shape_arr = (c_size_t * len(shape))()
    for i, s in enumerate(shape):
        shape_arr[i] = s

    tensor_ptr = LIB_LLAISYS.tensorCreate(
        shape_arr,
        len(shape),
        dtype.value if hasattr(dtype, 'value') else dtype,
        device.value if hasattr(device, 'value') else device,
        0  # device_id
    )

    return Tensor(tensor_ptr)
