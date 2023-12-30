import torch
from torch import einsum
import unittest
from slowgrad.engine import SlowgradVar
from typing import List, Tuple, Any, Literal
from slowgrad.functional import slowgrad_einsum, slowgrad_sigmoid, slowgrad_mse


class TestGradient(unittest.TestCase):
    def make_values(self, shapes: List[Tuple]) -> Tuple:
        torch_vals = [torch.randn(s, requires_grad=True) for s in shapes]
        slowgrad_vals = [SlowgradVar(x.detach().clone()) for x in torch_vals]
        return (torch_vals, slowgrad_vals)

    def test_scalar_addition_gradient(self):
        t = torch.randn(1, requires_grad=True)
        s = SlowgradVar(t)
        (t + t).backward()
        (s + s).backward()
        self.assertTrue(torch.allclose(s.grad, t.grad))

    # ! Write a multiplication method?
    def test_matrix_addition_gradient(self):
        (t, a), (s, b) = self.make_values([(5, 6), (5, 6)])
        (t + a).sum().backward()
        (s + b).sum().backward()
        self.assertTrue(torch.allclose(t.grad, s.grad))
        self.assertTrue(torch.allclose(a.grad, b.grad))

    def test_matrix_multiplication_gradient(self):
        (t, a), (s, b) = self.make_values([(2, 3), (3, 4)])
        ptrn = "ij,jk->"
        einsum(ptrn, t, a).backward()
        slowgrad_einsum(ptrn, s, b).sum().backward()
        self.assertTrue(torch.allclose(t.grad, s.grad))
        self.assertTrue(torch.allclose(a.grad, b.grad))

    def test_kronecker_gradient(self):
        (t, a), (s, b) = self.make_values([(2, 3), (3, 4)])
        ptrn = "ab,cd->abcd"
        einsum(ptrn, t, a).sum().backward()
        slowgrad_einsum(ptrn, s, b).sum().backward()
        self.assertTrue(torch.allclose(t.grad, s.grad))
        self.assertTrue(torch.allclose(a.grad, b.grad))

    def test_haddamard_gradient(self):
        (t, a), (s, b) = self.make_values([(2, 3), (2, 3)])
        ptrn = "ab,ab->"
        einsum(ptrn, t, a).backward()
        slowgrad_einsum(ptrn, s, b).backward()
        self.assertTrue(torch.allclose(t.grad, s.grad))
        self.assertTrue(torch.allclose(a.grad, b.grad))

    def test_sigmoid_gradient(self):
        (t1, t2, t3, t4), (s1, s2, s3, s4) = self.make_values(
            [(1,), (1, 1), (2, 2), (1, 2, 3)]
        )

        ft = lambda x: torch.sigmoid(x)
        fs = lambda x: slowgrad_sigmoid(x)

        for t, s in zip((t1, t2, t3, t4), (s1, s2, s3, s4)):
            ft(t).sum().backward()
            fs(s).sum().backward()
            self.assertTrue(torch.allclose(t1.grad, s1.grad))

    def test_mse_gradient(self):
        t_inputs, s_inputs = self.make_values([(1,), (1, 1), (2, 3), (1, 2, 3)])
        t_targets, s_targets = self.make_values([(1,), (1, 1), (2, 3), (1, 2, 3)])

        for t_input, s_input, t_target, s_target in zip(
            t_inputs, s_inputs, t_targets, s_targets
        ):
            torch.nn.MSELoss()(t_input, t_target).backward()
            slowgrad_mse(s_input, s_target).backward()
            self.assertTrue(torch.allclose(t_input.grad, s_input.grad))
            self.assertTrue(torch.allclose(t_target.grad, s_target.grad))

    def test_multiple_matrix_composition(self):
        t_matricies, s_matricies = self.make_values(
            [(2, 3), (3, 8), (8, 1), (1, 5), (2, 2), (3, 3)]
        )
        p1 = "ij,jk->ik"
        p2 = p1
        p3 = p1
        p4 = "ab,cd->cd"
        p5 = "ab,cd->"
        ptrns = [p1, p2, p3, p4, p5]

        t_out, s_out = t_matricies.pop(0), s_matricies.pop(0)
        for t, s, p in zip(t_matricies, s_matricies, ptrns):
            t_out = einsum(p, t_out, t)
            s_out = slowgrad_einsum(p, s_out, s)
        s_out.backward()
        t_out.backward()

        for t, s in zip(t_matricies, s_matricies):
            self.assertTrue(torch.allclose(t.grad, s.grad))
