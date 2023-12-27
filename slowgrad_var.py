import torch
from torch import einsum
from jacobian_operations import (
    create_backprop_einsum_pattern,
    compute_einsum_jacobian,
    swap_einsum_inputs,
    sigmoid_jacobian,
)


class SlowGradVar:
    """Encapsulates a tensor with automatic gradient"""

    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data
        self.grad: torch.Tensor = torch.zeros_like(self.data)

        self.local_jacobian: torch.Tensor = torch.Tensor()
        self.jacobian: torch.Tensor = torch.Tensor()

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def sum(self):
        """Return sum as a new SlowGradVar"""
        out = SlowGradVar(self.data.sum(), _children=(self,))

        def _backward():
            self.jacobian = torch.ones_like(self.data)

        out._backward = _backward
        return out

    def backward(self):
        """Backpropogates by calling the _backward function of each member in the computational graph"""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.jacobian = torch.tensor(1.0)
        for v in reversed(topo):
            v._backward()


def backpropogate(a: SlowGradVar, out: SlowGradVar) -> None:
    """Backpropogates from 'out' into a"""
    a.jacobian = einsum(
        create_backprop_einsum_pattern(out.jacobian.dim(), a.local_jacobian.dim()),
        out.jacobian,
        a.local_jacobian,
    )


def slowgrad_einsum(ptrn: str, a: SlowGradVar, b: SlowGradVar) -> SlowGradVar:
    """Preforms einsum operations and returns as a new SlowGradVar"""
    out = SlowGradVar(einsum(ptrn, a.data, b.data), _children=(a, b), _op="ein")

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


def slowgrad_sigmoid(a: SlowGradVar) -> SlowGradVar:
    """Applies sigmoid returning the result as a new SlowGradVar"""
    out = SlowGradVar(torch.sigmoid(a.data), _children=(a,), _op="sig")

    def _backward():
        a.local_jacobian = sigmoid_jacobian(a.data)
        backpropogate(a, out)

    out._backward = _backward
    return out
