import torch
from torch import einsum
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from .engine import SlowgradVar, backpropogate
from .autograd.jacobian import (
    compute_einsum_jacobian,
    swap_einsum_inputs,
    sigmoid_jacobian,
    mse_jacobian,
    softmax_jacobian,
    cross_entropy_jacobian,
    haddamard_sum_ptrn,
    relu_jacobian,
)


def slowgrad_einsum(ptrn: str, a: SlowgradVar, b: SlowgradVar) -> SlowgradVar:
    """Preforms einsum operations and returns as a new SlowGradVar"""
    assert any(
        [x.data.dim() > 1 for x in (a, b)]
    ), "Does not support scalar multiplication"
    out = SlowgradVar(einsum(ptrn, a.data, b.data), _children=(a, b), _op="ein")

    def _backward():
        def calc_local_jacobian(ptrn, x, y):
            x.local_jacobian = compute_einsum_jacobian(ptrn, x.data, y.data)

        def execute_backprop(ptrn, x, y, out):
            calc_local_jacobian(ptrn, x, y)
            backpropogate(out, x)

        execute_backprop(ptrn, a, b, out)
        execute_backprop(swap_einsum_inputs(ptrn), b, a, out)

    out._backward = _backward
    return out


def slowgrad_sigmoid(x: SlowgradVar) -> SlowgradVar:
    """Applies sigmoid returning the result as a new SlowGradVar"""
    out = SlowgradVar(torch.sigmoid(x.data), _children=(x,), _op="sig")

    def _backward():
        x.local_jacobian = sigmoid_jacobian(x.data)
        backpropogate(out, x)

    out._backward = _backward
    return out


def slowgrad_relu(x: SlowgradVar) -> SlowgradVar:
    out = SlowgradVar(torch.nn.ReLU()(x.data), _children=(x,))

    def _backward():
        x.local_jacobian = relu_jacobian(x.data)
        backpropogate(out, x)

    out._backward = _backward
    return out


def slowgrad_mse(x: SlowgradVar, y: SlowgradVar) -> SlowgradVar:
    out = SlowgradVar(((x.data - y.data) ** 2 / x.data.numel()).sum(), _children=(x, y))

    def _backward():
        x.local_jacobian = mse_jacobian(x.data, y.data)
        y.local_jacobian = mse_jacobian(y.data, x.data)
        backpropogate(out, x)
        backpropogate(out, y)

    out._backward = _backward
    return out


def slowgrad_softmax(x: SlowgradVar, dim: int = -1) -> SlowgradVar:
    out = SlowgradVar(F.softmax(x.data, dim=dim), _children=(x,))

    def _backward():
        x.local_jacobian = softmax_jacobian(x.data, dim=dim)
        backpropogate(out, x)

    out._backward = _backward
    return out


def slowgrad_cross_entropy_loss(x: SlowgradVar, y: SlowgradVar) -> SlowgradVar:
    dim = -1
    soft = F.softmax(x.data, dim=dim)
    log_soft = torch.log(soft)
    loss = -einsum(haddamard_sum_ptrn(x.data.dim()), log_soft, y.data) / x.data.shape[0]
    out = SlowgradVar(loss, _children=(x, y))

    def _backward():
        x.local_jacobian = cross_entropy_jacobian(x.data, y.data, soft, log_soft)
        backpropogate(out, x)

    out._backward = _backward
    return out
