from ctypes import POINTER, Structure, c_int32, c_void_p, cast, c_float
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)


def clip(x, min, max):
    return torch.clip(x, min, max)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    min,
    max,
    x_min,
    x_max,
    tensor_dtype=torch.float16,
):
    print(f"testing shape: {x_shape} min: {min} max: {max} tensor_dtype: {tensor_dtype}")
    x = (x_max - x_min) * torch.rand(x_shape, dtype=tensor_dtype).to(torch_device) + x_min
    min_scalar = torch.tensor(min, dtype=tensor_dtype).to(torch_device)
    max_scalar = torch.tensor(max, dtype=tensor_dtype).to(torch_device)
    y = (x_max - x_min) * torch.rand(x_shape, dtype=tensor_dtype).to(torch_device) + x_min

    ans = clip(x, min_scalar, max_scalar)
    x_tensor = to_tensor(x, lib)
    min_tensor = to_tensor(min_scalar, lib)
    max_tensor = to_tensor(max_scalar, lib)

    # 打印 max_tensor 的值
    max_tensor_value = cast(max_tensor.data, POINTER(c_float)).contents.value
    print(f"max_tensor: {max_tensor_value}")

    # 打印 min_tensor 的值
    min_tensor_value = cast(min_tensor.data, POINTER(c_float)).contents.value
    print(f"min_tensor: {min_tensor_value}")

    y_tensor = to_tensor(y, lib)

    descriptor = infiniopClipDescriptor_t()

    print(f"creating clip descriptor")
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            x_tensor.descriptor,
            min_tensor.descriptor,
            max_tensor.descriptor,
            y_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    min_tensor.descriptor.contents.invalidate()
    max_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    print(f"computing clip...")
    check_error(
        lib.infiniopClip(descriptor, x_tensor.data, min_tensor.data, max_tensor.data, y_tensor.data, None)
    )

    assert torch.allclose(y, ans, atol=0, rtol=0)
    print(f"clip passed")
    print(f"destroying clip descriptor")
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, min, max, x_min, x_max in test_cases:
        test(lib, handle, "cpu", x_shape, min, max, x_min, x_max, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, min, max, x_min, x_max, tensor_dtype=torch.float32)

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # (x_shape, min, max, x_min, x_max)
        # cross
        ((1,), 0.0, 4.0, -1.0, 3.0),
        ((100, 100), 0.0, 4.0, -1.0, 3.0),
        ((1, 1, 1), 0.0, 4.0, -1.0, 3.0),
        ((10, 10, 10), 0.0, 4.0, -1.0, 3.0),
        # inner bounds
        ((1,), 0.0, 2.0, -1.0, 3.0), 
        ((100, 100), 0.0, 2.0, -1.0, 3.0),
        ((1, 1, 1), 0.0, 2.0, -1.0, 3.0),
        ((10, 10, 10), 0.0, 2.0, -1.0, 3.0),
        # outer bounds
        ((1,), -3.0, 5.0, -1.0, 3.0),
        ((100, 100), -3.0, 5.0, -1.0, 3.0),
        ((1, 1, 1), -3.0, 5.0, -1.0, 3.0),
        ((10, 10, 10), -3.0, 5.0, -1.0, 3.0),
        # min greater than max
        ((1,2,3), 2.0, 0.0, -1.0, 3.0),
        # # default
        # ((1,2,3), 0, torch.inf, -1.0, 3.0),
        # ((1,2,3), -torch.inf, 2.0, -1.0, 3.0)
        # large inputs
        ((1024,), 0.0, 2.0, -1.0, 3.0),
        ((100, 100), 0.0, 2.0, -1.0, 3.0),
        ((32, 12, 4), 0.0, 2.0, -1.0, 3.0),
        ((64, 32, 32), 0.0, 2.0, -1.0, 3.0),
    ]

    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)

    print("\033[92mTest passed!\033[0m")