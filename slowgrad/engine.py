import torch
from torch import einsum
import string
from .autograd.jacobian import create_backprop_einsum_pattern


class SlowgradVar:
    """Encapsulates a tensor with automatic gradient"""

    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        self.grad: torch.Tensor = torch.zeros_like(self.data)

        self.local_jacobian: torch.Tensor = torch.Tensor()
        self.jacobian: torch.Tensor = torch.Tensor()

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, SlowgradVar) else SlowgradVar(other)
        out = SlowgradVar(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.jacobian, other.jacobian = out.jacobian, out.jacobian

            def broadcast_reshape(x, j):
                trailing_dims_length = sum(
                    1 for x in zip(reversed(x.shape), reversed(j.shape)) if x[0] == x[1]
                )
                sum_ptrn = string.ascii_lowercase[:trailing_dims_length]
                x += einsum(f"...{sum_ptrn}->{sum_ptrn}", j)

            def update_gradient(x, j):
                if x.shape == j.shape:
                    x += j
                else:
                    broadcast_reshape(x, j)

            update_gradient(self.grad, out.jacobian)
            update_gradient(other.grad, out.jacobian)

        out._backward = _backward
        return out

    def sum(self):
        """Return sum as a new SlowGradVar"""
        out = SlowgradVar(self.data.sum(), _children=(self,))

        def _backward():
            # ! Theres somethign fishy here....
            # ? Ok I think we are good here.
            self.local_jacobian = torch.ones_like(self.data)
            backpropogate(self, out)
            self.grad += self.jacobian

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
        # ! Check this
        self.jacobian = torch.tensor(1.0)
        for v in reversed(topo):
            v._backward()


def backpropogate(a: SlowgradVar, out: SlowgradVar) -> None:
    """Backpropogates from 'out' into a"""
    # print(a.jacobian.dim(), out.jacobian.shape)
    # print(create_backprop_einsum_pattern(out.jacobian.dim(), a.local_jacobian.dim()))
    a.jacobian = einsum(
        create_backprop_einsum_pattern(out.jacobian.dim(), a.local_jacobian.dim()),
        out.jacobian,
        a.local_jacobian,
    )