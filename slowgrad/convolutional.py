import torch
from .engine import SlowgradVar, backpropogate
from .functional import slowgrad_einsum
from torch import einsum


def make_convolution_windows(
    x: torch.Tensor, kernel_size: int, stride: int
) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1

    windowed_input_tensor = torch.as_strided(
        x,
        size=(
            batch_size,
            channels,
            output_height,
            output_width,
            kernel_size,
            kernel_size,
        ),
        stride=(
            x.stride(0),
            x.stride(1),
            x.stride(2) * stride,
            x.stride(3) * stride,
            x.stride(2),
            x.stride(3),
        ),
    )
    return windowed_input_tensor


def invert_convolution_windows(windowed_tensor, original_shape, kernel_size, stride):
    batch_size, channels, height, width = original_shape
    accumulator = torch.zeros(original_shape)
    # ! get rid of these gross ass for loops
    for i in range(0, height - kernel_size + 1, stride):
        for j in range(0, width - kernel_size + 1, stride):
            accumulator[
                :, :, i : i + kernel_size, j : j + kernel_size
            ] += windowed_tensor[:, :, i // stride, j // stride, :, :]

    return accumulator


def slowgrad_2d_convolution(
    x: SlowgradVar, w: SlowgradVar, kernel_size: int, stride: int
) -> SlowgradVar:
    windows = SlowgradVar(make_convolution_windows(x.data, kernel_size, stride))
    out = slowgrad_einsum("bchwkt,fckt->bfhw", windows, w)

    out._prev = set((x, w))  # windows is temp, so we remove it from comp graph

    def _backward():
        dx = einsum("bfhw,fckt->bchwkt", torch.ones_like(out.data), w.data)
        dx = einsum("bchwkt,bfhw->bchwkt", dx, out.jacobian)
        dx = invert_convolution_windows(dx, x.data.shape, kernel_size, stride)
        x.jacobian = dx
        x.grad += dx

    out._backward = _backward
    return out
