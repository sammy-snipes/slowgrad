import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from slowgrad.engine import SlowgradVar
from slowgrad.models import (
    SlowgradSequential,
    SlowgradMSELoss,
    SlowgradCrossEntropyLoss,
)
from einops import rearrange


class TestActivations(unittest.TestCase):
    def reshape_torch_gradient_tensor(self, x) -> torch.Tensor:
        # in torch x.grad is in the shape of x.T,
        # .T is deprecated, hence this
        if x.dim() == 1:
            return rearrange(x, "a -> 1 a")
        else:
            return rearrange(x, "a b -> b a")

    def compare_parameter_gradients(self, t_params, s_params):
        return [
            torch.allclose(self.reshape_torch_gradient_tensor(pt.grad), ps.grad)
            for (pt, ps) in zip(t_params, s_params)
        ]

    def make_dummy_network(
        self,
        in_channels: int,
        out_channels: int,
    ):
        model = nn.Sequential(
            nn.Linear(in_channels, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, out_channels),
        )
        return (model, SlowgradSequential().from_torch(model))

    def test_mse_loss(self):
        batch_size, in_channels, out_channels = 10, 8, 8
        x, y = torch.randn(batch_size, in_channels), torch.randn(
            batch_size, out_channels
        )
        s, q = SlowgradVar(x), SlowgradVar(y)

        t_model, s_model = self.make_dummy_network(in_channels, out_channels)

        t_loss = nn.MSELoss()(t_model(x), y)
        s_loss = SlowgradMSELoss()(s_model(s), q)
        t_loss.backward()
        s_loss.backward()
        self.assertTrue(torch.allclose(t_loss, s_loss.data))
        self.assertTrue(
            all(
                self.compare_parameter_gradients(
                    t_model.parameters(), s_model.parameters()
                )
            )
        )

    def test_cross_entropy_loss(self):
        batch_size, in_channels, out_channels = 10, 8, 4
        x = torch.randn(batch_size, in_channels)
        y = F.one_hot(
            torch.randint(0, out_channels, (batch_size,)), num_classes=out_channels
        ).float()

        s, q = SlowgradVar(x), SlowgradVar(y)

        t_model, s_model = self.make_dummy_network(in_channels, out_channels)

        t_loss = nn.CrossEntropyLoss()(t_model(x), y)
        s_loss = SlowgradCrossEntropyLoss()(s_model(s), q)
        t_loss.backward()
        s_loss.backward()
        self.assertTrue(torch.allclose(t_loss, s_loss.data))
        self.assertTrue(
            all(
                self.compare_parameter_gradients(
                    t_model.parameters(), s_model.parameters()
                )
            )
        )
