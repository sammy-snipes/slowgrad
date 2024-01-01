from typing import Any, List, Optional
from .engine import SlowgradVar
from .functional import (
    slowgrad_einsum,
    slowgrad_sigmoid,
    slowgrad_mse,
    slowgrad_softmax,
)
import torch
import torch.nn as nn


class SlowgradModule:
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> Any:
        pass

    def from_torch(self, x) -> "SlowgradModule":
        return self

    def parameters(self):
        return []


class SlowgradLinear(SlowgradModule):
    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: Optional[bool] = None,
    ) -> None:
        self.weight = (
            SlowgradVar(torch.randn(in_features, out_features))
            if in_features and out_features
            else None
        )
        self.bias = (
            SlowgradVar(torch.randn(1, out_features)) if bias and out_features else None
        )

    def __call__(self, x: SlowgradVar) -> SlowgradVar:
        out = slowgrad_einsum("ij,jk->ik", x, self.weight)
        if self.bias:
            out += self.bias
        return out

    def from_torch(self, x: nn.Linear) -> "SlowgradLinear":
        self.weight = SlowgradVar(x.weight.detach().clone().T)
        if x.bias is not None:
            self.bias = SlowgradVar(x.bias.detach().clone().unsqueeze(0))
        else:
            self.bias = None
        return self

    def parameters(self) -> List[SlowgradVar]:
        return [p for p in (self.weight, self.bias) if p is not None]


class SlowgradSigmoid(SlowgradModule):
    def __call__(self, x: SlowgradVar) -> SlowgradVar:
        return slowgrad_sigmoid(x)


class SlowgradMSELoss(SlowgradModule):
    def __call__(self, input, target) -> SlowgradVar:
        return slowgrad_mse(input, target)


class SlowgradSoftmax(SlowgradModule):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x) -> SlowgradVar:
        return slowgrad_softmax(x, self.dim)


class SlowgradSequential:
    def __init__(self, layers: Optional[List[SlowgradModule]] = None) -> None:
        if layers:
            non_slowgrad_modules = [
                l for l in layers if not isinstance(l, SlowgradModule)
            ]
            assert (
                not non_slowgrad_modules
            ), f"{non_slowgrad_modules} are not slowgrad modules"
            self.layers = layers

    def from_torch(self, layers: nn.Sequential):
        implemented_layers = {
            nn.Sigmoid: SlowgradSigmoid,
            nn.Linear: SlowgradLinear,
            nn.MSELoss: SlowgradMSELoss,
        }
        missing_implementation = [
            type(l) for l in layers if type(l) not in implemented_layers
        ]
        assert (
            not missing_implementation
        ), f"missing implemntation for : {missing_implementation}"

        self.layers = [implemented_layers[type(l)]().from_torch(l) for l in layers]
        return self

    def __call__(self, x: SlowgradVar) -> SlowgradVar:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> List[SlowgradVar]:
        params = [l.parameters() for l in self.layers]
        return sum(params, [])
