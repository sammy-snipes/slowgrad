import torch
import torch.nn as nn
import unittest
from slowgrad.engine import SlowgradVar
from slowgrad.models import SlowgradSequential
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
            torch.allclose(
                self.reshape_torch_gradient_tensor(pt.grad), ps.grad, atol=1e-5
            )
            for (pt, ps) in zip(t_params, s_params)
        ]

    def make_dummy_network(
        self,
        in_channels: int,
        out_channels: int,
        activation,
    ):
        model = nn.Sequential(
            nn.Linear(in_channels, 8, bias=True),
            activation(),
            nn.Linear(8, 8, bias=True),
            activation(),
            nn.Linear(8, out_channels, bias=True),
        )
        return (model, SlowgradSequential().from_torch(model))

    def test_sigmoid(self):
        batch_size, in_channels, out_channels = 10, 8, 8
        x = torch.randn(batch_size, in_channels)
        s = SlowgradVar(x)
        t_model, s_model = self.make_dummy_network(
            in_channels, out_channels, activation=nn.Sigmoid
        )

        t_out = t_model(x).sum()
        s_out = s_model(s).sum()
        t_out.backward()
        s_out.backward()
        self.assertTrue(torch.allclose(t_out, s_out.data))
        self.assertTrue(
            all(
                self.compare_parameter_gradients(
                    t_model.parameters(), s_model.parameters()
                )
            )
        )

    def test_relu(self):
        batch_size, in_channels, out_channels = 10, 80, 1
        x = torch.randn(batch_size, in_channels)
        s = SlowgradVar(x)
        t_model, s_model = self.make_dummy_network(
            in_channels, out_channels, activation=nn.ReLU
        )

        t_out = t_model(x).sum()
        s_out = s_model(s).sum()
        t_out.backward()
        s_out.backward()

        print(
            self.compare_parameter_gradients(t_model.parameters(), s_model.parameters())
        )
        print(
            self.reshape_torch_gradient_tensor(list(t_model.parameters())[0]),
            s_model.parameters()[0].data,
            sep=2 * "\n",
        )

        self.assertTrue(torch.allclose(t_out, s_out.data))
        self.assertTrue(
            all(
                self.compare_parameter_gradients(
                    t_model.parameters(), s_model.parameters()
                )
            )
        )
