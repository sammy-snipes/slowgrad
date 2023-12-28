from .engine import SlowgradVar

from .jacobian_functions import (
    compute_einsum_jacobian,
    create_backprop_einsum_pattern,
    swap_einsum_inputs,
    sigmoid_jacobian,
)
import torch
from torch import einsum
from einops import repeat
import torch.nn as nn


def slowgrad_einsum(ptrn: str, a: SlowgradVar, b: SlowgradVar) -> SlowgradVar:
    """Preforms einsum operations and returns as a new SlowGradVar"""
    out = SlowgradVar(einsum(ptrn, a.data, b.data), _children=(a, b), _op="ein")

    def _backward():
        def calc_local_jacobian(ptrn, x, y):
            x.local_jacobian = compute_einsum_jacobian(ptrn, x.data, y.data)

        def update_gradient(x):
            x.grad += x.jacobian

        def execute_backprop(ptrn, x, y, out):
            calc_local_jacobian(ptrn, x, y)
            backpropogate(x, out)
            update_gradient(x)

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


def backpropogate(a: SlowgradVar, out: SlowgradVar) -> None:
    """Backpropogates from 'out' into a"""
    a.jacobian = einsum(
        create_backprop_einsum_pattern(out.jacobian.dim(), a.local_jacobian.dim()),
        out.jacobian,
        a.local_jacobian,
    )


class SlowgradLinear:
    def __init__(self, in_features=None, out_features=None, bias=None) -> None:
        self.weight = (
            SlowgradVar(torch.randn(in_features, out_features))
            if in_features and out_features
            else None
        )
        self.bias = (
            SlowgradVar(torch.randn(1, out_features)) if bias and out_features else None
        )

    def __call__(self, x: SlowgradVar):
        out = slowgrad_einsum("ij,jk->ik", x, self.weight)
        if self.bias:
            out += self.bias
        return out

    def from_torch(self, x: nn.Linear) -> "SlowgradLinear":
        self.weight = SlowgradVar(x.weight.detach().clone().T)
        if x.bias is not None:
            self.bias = SlowgradVar(x.bias.detach().clone().unsqueeze(0))
        else:
            self.bias = None
        return self
