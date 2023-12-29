import torch
from torch import einsum
import unittest
from slowgrad.engine import SlowgradVar
from typing import List, Tuple, Any


class TestGradient(unittest.TestCase):
    def make_values(self, shapes: List[Tuple[int]]) -> Tuple:
        torch_vals = [torch.randn(s, requires_grad=True) for s in shapes]
        slowgrad_vals = [SlowgradVar(x.detach().clone()) for x in torch_vals]
        return (torch_vals, slowgrad_vals)

    def test_addition_gradient(self):
        (xt, yt), (xs, ys) = self.make_values([(1,), (1,)])
        zt = xt + yt
        zs = xs + ys
        zt.backward()
        zs.backward()
        self.assertTrue(torch.allclose(xt.grad, xs.grad))
        self.assertTrue(torch.allclose(yt.grad, ys.grad))

    def test_multiple_addition_gradient(self):
        (xt, yt), (xs, ys) = self.make_values([(1,), (1,)])
        zt = xt + xt + xt + yt + yt
        zs = xs + xs + xs + ys + ys
        zt.backward()
        zs.backward()
        self.assertTrue(torch.allclose(xt.grad, xs.grad))
        self.assertTrue(torch.allclose(yt.grad, ys.grad))

    def test_matrix_addition_gradient(self):
        pass

    def test_multiple_matrix_addition_gradient(self):
        pass

    def test_broadcasting_addition_gradient(self):
        pass

    def test_einsum_mm_gradient(self):
        pass

    def test_einsum_kronecker_gradient(self):
        pass

    def test_einsum_bmm_gradient(self):
        pass

    def test_einsum_haddamard_gradient(self):
        pass

    def test_einsum_idk_gradient(self):
        pass

    def test_sigmoid_sum_gradient(self):
        pass

    def test_sigmoid_einsum_composition_gradient(self):
        pass

    def test_multiple_sigmoid_einsum_composition_gradient(self):
        pass

    def test_mse_gradient(self):
        pass
