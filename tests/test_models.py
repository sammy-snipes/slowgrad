from slowgrad.models import SlowgradSequential, SlowgradMSELoss
from slowgrad.engine import SlowgradVar
import torch
import torch.nn as nn
import unittest
from einops import rearrange


class TestSequential(unittest.TestCase):
    def reshape_torch_gradient_tensor(self, x) -> torch.Tensor:
        # in torch x.grad is in the shape of x.T,
        # .T is deprecated, hence this
        if x.dim() == 1:
            return rearrange(x, "a -> 1 a")
        else:
            return rearrange(x, "a b -> b a")

    def compare_params(self, t_params, s_params):
        return [
            torch.allclose(self.reshape_torch_gradient_tensor(pt.grad), ps.grad)
            for (pt, ps) in zip(t_params, s_params)
        ]

    def test_sigmoid_nn_sum_loss(self):
        t = torch.randn(2, 5, requires_grad=True)
        s = SlowgradVar(t)

        torch_model = nn.Sequential(
            nn.Linear(5, 5, bias=True),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=True),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 1),
        )
        slowgrad_model = SlowgradSequential().from_torch(torch_model)

        torch_model(t).sum().backward()
        slowgrad_model(s).sum().backward()

        self.assertTrue(
            all(
                self.compare_params(
                    torch_model.parameters(), slowgrad_model.parameters()
                )
            )
        )

    def test_sigmoid_nn_mse_loss(self):
        t, a = torch.randn(2, 5, requires_grad=True), torch.randn(2, 1)
        s, b = SlowgradVar(t), SlowgradVar(a)

        torch_model = nn.Sequential(
            nn.Linear(5, 5, bias=True),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=True),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )
        slowgrad_model = SlowgradSequential().from_torch(torch_model)

        nn.MSELoss()(torch_model(t), a).backward()
        SlowgradMSELoss()(slowgrad_model(s), b).backward()

        self.assertTrue(
            all(
                self.compare_params(
                    torch_model.parameters(), slowgrad_model.parameters()
                )
            )
        )
