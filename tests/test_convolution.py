import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from slowgrad.engine import SlowgradVar
from slowgrad.models import SlowgradSequential, SlowgradConv2d, SlowgradCrossEntropyLoss
from slowgrad.functional import slowgrad_einsum
from einops import rearrange
from torch import einsum


class TestConv2dOutputEquality(unittest.TestCase):
    def test_one_conv(self):
        batch_size, in_channels, height, width = 1, 3, 4, 4
        out_channels, kernel_size, stride = 1, 2, 1

        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        s = SlowgradVar(x)

        torch_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, bias=False
        )
        slowgrad_conv = SlowgradConv2d().from_torch(torch_conv)

        torch_out = torch_conv(x)
        slowgrad_out = slowgrad_conv(s)

        self.assertTrue(torch.allclose(torch_out, slowgrad_out.data))

    def test_multiple_conv(self):
        batch_size, in_channels, height, width = 1, 3, 128, 25
        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        s = SlowgradVar(x)

        torch_conv1 = nn.Conv2d(
            in_channels, out_channels=4, kernel_size=2, stride=1, bias=False
        )
        torch_conv2 = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=6, stride=3, bias=False
        )
        torch_conv3 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=2, stride=2, bias=False
        )
        torch_conv4 = nn.Conv2d(
            in_channels=6, out_channels=3, kernel_size=3, stride=1, bias=False
        )

        torch_convs = [torch_conv1, torch_conv2, torch_conv3, torch_conv4]
        slowgrad_convs = [SlowgradConv2d().from_torch(_) for _ in torch_convs]

        out = [(x, s)]
        for t_layer, s_layer in zip(torch_convs, slowgrad_convs):
            t_out, s_out = out[-1]
            out.append((t_layer(t_out), s_layer(s_out)))

        # for t_out, s_out in out:
        #     abs_diff = (t_out - s_out.data).sum()
        #     print(abs_diff, t_out.shape)

        # ! Floating point weirdness...
        self.assertTrue(
            all(
                [torch.allclose(t_out, s_out.data, atol=1e-5) for (t_out, s_out) in out]
            )
        )


class TestConv2dGradient(unittest.TestCase):
    def compare_parameter_gradients(self, t_params, s_params):
        return [
            torch.allclose(pt.grad, ps.grad) for (pt, ps) in zip(t_params, s_params)
        ]

    def test_one_conv_gradient(self):
        batch_size, in_channels, height, width = 1, 3, 4, 4
        out_channels, kernel_size, stride = 1, 2, 1

        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        s = SlowgradVar(x)

        torch_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, bias=False
        )
        slowgrad_conv = SlowgradConv2d().from_torch(torch_conv)

        torch_conv(x).sum().backward()
        slowgrad_conv(s).sum().backward()

        self.assertTrue(torch.allclose(x.grad, s.grad))

    def test_multiple_conv_gradient(self):
        batch_size, in_channels, height, width = 1, 3, 40, 40
        out_channels, kernel_size, stride = 1, 2, 1

        torch_conv1 = nn.Conv2d(
            in_channels, out_channels=4, kernel_size=2, stride=1, bias=False
        )
        torch_conv2 = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, stride=2, bias=False
        )
        torch_conv3 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=2, stride=1, bias=False
        )
        torch_conv4 = nn.Conv2d(
            in_channels=6, out_channels=3, kernel_size=3, stride=1, bias=False
        )

        torch_convs = [torch_conv1, torch_conv2, torch_conv3, torch_conv4]
        slowgrad_convs = [SlowgradConv2d().from_torch(_) for _ in torch_convs]

        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        s = SlowgradVar(x)

        out = [(x, s)]
        for t_layer, s_layer in zip(torch_convs, slowgrad_convs):
            t_out, s_out = out[-1]
            out.append((t_layer(t_out), s_layer(s_out)))

        out[-1][0].sum().backward()
        out[-1][1].sum().backward()

        t_params = [l.weight for l in torch_convs]
        s_params = [l.weight for l in slowgrad_convs]

        self.compare_parameter_gradients(t_params, s_params)

    def test_multiple_conv_grad_with_shenanigans(self):
        batch_size, in_channels, height, width = 5, 3, 8, 8

        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        s = SlowgradVar(x)

        torch_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=4, kernel_size=2, stride=1, bias=False
            ),
            nn.Sigmoid(),
            nn.Conv2d(
                in_channels=4, out_channels=4, kernel_size=2, stride=1, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=4, out_channels=4, kernel_size=2, stride=1, bias=False
            ),
            nn.Softmax(dim=-1),
        )
        slowgrad_model = SlowgradSequential().from_torch(torch_model)

        torch_out = einsum("abcd->ad", torch_model(x))
        slowgrad_out = slowgrad_model(s)
        slowgrad_out = slowgrad_einsum(
            "abcd,abcd->ad",
            slowgrad_out,
            SlowgradVar(torch.ones_like(slowgrad_out.data)),
        )

        t_targets = F.one_hot(
            torch.randint(0, torch_out.shape[-1], (batch_size,)),
            num_classes=torch_out.shape[-1],
        ).float()

        s_targets = SlowgradVar(t_targets)
        nn.CrossEntropyLoss()(torch_out, t_targets).backward()
        SlowgradCrossEntropyLoss()(slowgrad_out, s_targets).backward()

        self.compare_parameter_gradients(
            torch_model.parameters(), slowgrad_model.parameters()
        )
