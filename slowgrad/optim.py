from .engine import SlowgradVar
from typing import List, Tuple
import torch


# ! Write esomething else besides adam...normal SGD fine too..
class SlowgradAdam:
    def __init__(
        self,
        params: List[SlowgradVar],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {p: torch.zeros_like(p.data) for p in params}
        self.v = {p: torch.zeros_like(p.data) for p in params}
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            # ! Is this okay?
            p.local_jacobian = torch.zeros_like(p.local_jacobian)
            p.jacobian = torch.zeros_like(p.jacobian)
            p.grad = torch.zeros_like(p.grad)

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue

            if self.weight_decay != 0:
                param.grad += self.weight_decay * param.data

            self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * (
                param.grad
            )
            self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (
                param.grad**2
            )

            m_hat = self.m[param] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[param] / (1 - self.betas[1] ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
