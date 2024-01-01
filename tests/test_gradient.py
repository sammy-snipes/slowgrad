import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import unittest
from slowgrad.engine import SlowgradVar
from typing import List, Tuple, Any, Literal
from slowgrad.functional import (
    slowgrad_einsum,
    slowgrad_sigmoid,
    slowgrad_mse,
    slowgrad_softmax,
    slowgrad_cross_entropy_loss,
)


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

    def test_softmax_gradient(self):
        atol = 1e-6
        # Softmax gradients are really small
        # the resulting gradients are 1e-6 close instead of 1e-8
        (t1, t2, t3, t4, t5), (s1, s2, s3, s4, s5) = self.make_values(
            [(1, 1), (2, 1), (10, 20), (1, 2, 3), (1, 2, 3, 1, 2, 3)]
        )

        for t, s in zip((t1, t2, t3, t4, t5), (s1, s2, s3, s4, s5)):
            for dim in [i for i in range(t.dim())] + [-i for i in range(t.dim())]:
                ft = lambda x: torch.nn.functional.softmax(x, dim=dim)
                fs = lambda x: slowgrad_softmax(x, dim=dim)

                ft(t).sum().backward()
                fs(s).sum().backward()
                self.assertTrue(torch.allclose(t.grad, s.grad, atol=atol))

    def test_cross_entropy_loss_gradient(self):
        def make_class_data(batch_size, num_classes):
            x = torch.randn(batch_size, num_classes, requires_grad=True)
            y = F.one_hot(
                torch.arange(0, batch_size) % num_classes, num_classes=num_classes
            ).type(torch.float)
            xs, ys = SlowgradVar(x.detach().clone()), SlowgradVar(y.detach().clone())
            return (x, xs), (y, ys)

        for shape in [(1, 1), (1, 2), (2, 2), (20, 2), (10, 100)]:
            (x, xs), (y, ys) = make_class_data(*shape)

            ft = lambda x: nn.CrossEntropyLoss()(x, y)
            fs = lambda xs: slowgrad_cross_entropy_loss(xs, ys)

            ft(x).backward()
            fs(xs).backward()
            self.assertTrue(torch.allclose(x.grad, xs.grad))
