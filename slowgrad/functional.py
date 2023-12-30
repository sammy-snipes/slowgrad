import torch
from torch import einsum
from .engine import SlowgradVar
from .autograd.jacobian import (
    compute_einsum_jacobian,
    swap_einsum_inputs,
    sigmoid_jacobian,
    create_backprop_einsum_pattern,
    mse_jacobian,
)


def backpropogate(a: SlowgradVar, out: SlowgradVar) -> None:
    """Backpropogates from 'out' into a"""
    a.jacobian = einsum(
        create_backprop_einsum_pattern(out.jacobian.dim(), a.local_jacobian.dim()),
        out.jacobian,
        a.local_jacobian,
    )
    a.grad += a.jacobian


def slowgrad_einsum(ptrn: str, a: SlowgradVar, b: SlowgradVar) -> SlowgradVar:
    """Preforms einsum operations and returns as a new SlowGradVar"""
    assert any(
        [x.data.dim() > 1 for x in (a, b)]
    ), "Does not support scalar multiplication"
    out = SlowgradVar(einsum(ptrn, a.data, b.data), _children=(a, b), _op="ein")

    def _backward():
        # ? I dont know how I feel about the logic of this. Maybe it would be better to have these
        # ? return the value, and then set it. or something

        def calc_local_jacobian(ptrn, x, y):
            x.local_jacobian = compute_einsum_jacobian(ptrn, x.data, y.data)

        def execute_backprop(ptrn, x, y, out):
            calc_local_jacobian(ptrn, x, y)
            backpropogate(x, out)

        execute_backprop(ptrn, a, b, out)
        execute_backprop(swap_einsum_inputs(ptrn), b, a, out)

    out._backward = _backward
    return out


def slowgrad_sigmoid(a: SlowgradVar) -> SlowgradVar:
    """Applies sigmoid returning the result as a new SlowGradVar"""
    out = SlowgradVar(torch.sigmoid(a.data), _children=(a,), _op="sig")

    def _backward():
        a.local_jacobian = sigmoid_jacobian(a.data)
        backpropogate(a, out)

    out._backward = _backward
    return out


def slowgrad_mse(x: SlowgradVar, y: SlowgradVar) -> SlowgradVar:
    out = SlowgradVar(((x.data - y.data) ** 2 / x.data.numel()).sum(), _children=(x, y))

    def _backward():
        x.local_jacobian = mse_jacobian(x.data, y.data)
        y.local_jacobian = mse_jacobian(y.data, x.data)
        backpropogate(x, out)
        backpropogate(y, out)

    out._backward = _backward
    return out
