import torch
from torch import einsum
import unittest
from slowgrad.autograd.jacobian import (
    mse_jacobian,
    sigmoid_jacobian,
    compute_einsum_jacobian,
    swap_einsum_inputs,
    softmax_jacobian,
)
from torch.autograd.functional import jacobian
from typing import List, Tuple, Callable
import torch.nn.functional as F


class TestEinsumJacobian(unittest.TestCase):
    def eval_compare(self, x, y, ptrn):
        x_ptrn, y_ptrn = ptrn, swap_einsum_inputs(ptrn)
        f_x, f_y = lambda x: einsum(x_ptrn, x, y), lambda y: einsum(y_ptrn, y, x)
        torch_x_jacobian, torch_y_jacobian = jacobian(f_x, x), jacobian(f_y, y)

        manual_x_jacobian = compute_einsum_jacobian(x_ptrn, x, y)
        manual_y_jacobian = compute_einsum_jacobian(y_ptrn, y, x)

        self.assertTrue(torch.allclose(torch_x_jacobian, manual_x_jacobian))
        self.assertTrue(torch.allclose(torch_y_jacobian, manual_y_jacobian))

    def test_mm(self):
        x, y = torch.randn(2, 3), torch.randn(3, 4)
        ptrn = "ij,jk->ik"
        self.eval_compare(x, y, ptrn)

    def test_bmm(self):
        x, y = torch.randn(3, 4, 5), torch.randn(3, 5, 10)
        ptrn = "bij,bjk->bik"
        self.eval_compare(x, y, ptrn)

    def test_kronecker(self):
        x, y = torch.randn(8, 8), torch.randn(2, 2)
        ptrn = "ab,cd->abcd"
        self.eval_compare(x, y, ptrn)

    def test_haddamard(self):
        x, y = torch.randn(5, 5), torch.randn(5, 5)
        ptrn = "ij,ij->ij"
        self.eval_compare(x, y, ptrn)

    def test_mm_sum(self):
        x, y = torch.randn(2, 3), torch.randn(3, 4)
        ptrn = "ij,jk->"
        self.eval_compare(x, y, ptrn)

    def test_mm_vec_sum(self):
        x, y = torch.randn(2, 3), torch.randn(3, 4)
        ptrn = "ij,jk->j"
        self.eval_compare(x, y, ptrn)

    def test_kronecker_sum(self):
        x, y = torch.randn(2, 3), torch.randn(3, 4)
        ptrn = "ab,cd->"
        self.eval_compare(x, y, ptrn)

    def test_idk(self):
        x, y = torch.randn(1, 2, 6, 4, 8), torch.randn(6, 7, 8)
        ptrn = "abcde,cze->ace"
        self.eval_compare(x, y, ptrn)


class TestSigmoidJacobian(unittest.TestCase):
    def eval_compare(self, x):
        f = lambda x: torch.sigmoid(x)
        torch_jacobian = jacobian(f, x)
        manual_jacobian = sigmoid_jacobian(x)
        self.assertTrue(torch.allclose(torch_jacobian, manual_jacobian))

    def test_scalar_sigmoid_jacobian(self):
        x = torch.randn(1)
        self.eval_compare(x)

    def test_matrix_sigmoid_jacobian(self):
        x = torch.randn(10, 15)
        self.eval_compare(x)

    def test_idk_matrix_sigmoid_jacobian(self):
        x = torch.randn(1, 1, 2, 3, 7)
        self.eval_compare(x)


class TestMSEJacobian(unittest.TestCase):
    def eval_compare(self, x, y):
        f_x = lambda x: torch.nn.MSELoss()(x, y)
        f_y = lambda y: torch.nn.MSELoss()(x, y)
        torch_x_jacobian = jacobian(f_x, x)
        torch_y_jacobian = jacobian(f_y, y)
        manual_x_jacobian = mse_jacobian(x, y)
        manual_y_jacobian = mse_jacobian(y, x)
        self.assertTrue(torch.allclose(torch_x_jacobian, manual_x_jacobian))
        self.assertTrue(torch.allclose(torch_y_jacobian, manual_y_jacobian))

    def test_scalar_mse(self):
        x, y = torch.randn(1), torch.randn(1)
        self.eval_compare(x, y)

    def test_matrix_mse(self):
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        self.eval_compare(x, y)

    def test_idk_matrix_mse(self):
        x, y = torch.randn(1, 2, 3, 4, 5, 6, 7), torch.randn(1, 2, 3, 4, 5, 6, 7)
        self.eval_compare(x, y)


class TestSoftmaxJacobian(unittest.TestCase):
    def eval_compare(self, x, dim):
        f = lambda x: F.softmax(x, dim)

        torch_jacobian = jacobian(f, x)
        manual_jacobian = softmax_jacobian(x, dim)
        self.assertTrue(torch.allclose(torch_jacobian, manual_jacobian))

    def test_softmax_jacobian_2d(self):
        x = torch.randn(5, 6, requires_grad=True)
        for dim in [0, 1, -1]:
            self.eval_compare(x, dim)

    def test_softmax_jacobian_3d(self):
        x = torch.randn(3, 5, 6, requires_grad=True)
        for dim in [0, 1, 2, -1, -2]:
            self.eval_compare(x, dim)

    def test_softmax_jacobian_big_d(self):
        x = torch.randn(1, 2, 3, 1, 2, 3, 1, 2, 3, requires_grad=True)
        for dim in [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]:
            self.eval_compare(x, dim)
