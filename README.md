# slowgrad

![](its_really_slow.png)

Autograd engine built around einsum (work in progress). Its really slow. Since NN's, CNN's, and even transformers can be written almost entirely with einsum, I thought it would be fun to write an autograd library that just implements the backward pass of einsum. Heavy inspiration taken from Andrej Kaparthy's [micrograd](https://github.com/karpathy/micrograd/tree/master)

The meat and potatoes are in `slowgrad/autograd/jacobian.py` which handles jacobian/gradient calculation, and then `slowgrad/engine.py` which is a wrapper for holding values. 150 and 100 lines of code, respectively. The rest of this junk is utils to make it feel more like pytorch.

### Example usage

```python
import torch
from slowgrad.engine import SlowgradVar
from slowgrad.models import (
    SlowgradSequential,
    SlowgradLinear,
    SlowgradSigmoid,
    SlowgradMSELoss,
)

model = SlowgradSequential([
    SlowgradLinear(2, 4, bias=True),
    SlowgradSigmoid(),
    SlowgradLinear(4, 4, bias=True),
    SlowgradSigmoid(),
    SlowgradLinear(4, 1, bias = True)
])

x, y = SlowgradVar(torch.randn(100, 2)), SlowgradVar(torch.randn(100, 1))

SlowgradMSELoss()(model(x), y).backward()
params = model.parameters()
for p in params:
    print(p.grad, '\n')

```

### How it works

Its helpful to think of `einsum('', x, y)` as working in two steps.

1. make some $(x.shape)\times(y.shape)$ matrix
2. sum over some of the axes

If we take the kronecker product of two 2D matricies $X, Y$ in einsum, we get a 4D tensor $K$ where `K[i, j, :, :]` $ = x_{i, j} * Y$

If do matrix multiplication we get a 4D tensor $K$ where

$$
K[i, j, k, l] =
\begin{cases}
x_{ij} * y_{kl} \ \ if  \  \ j = k \\
0 \ \ \ else
\end{cases}
$$

and then we sum over the inner two dimensions. In this framework we can find the jacobian of the expanded output of shape $(x.shape)\times(y.shape)$, and then sum over the necessary dimensions and were done.

Lets look at examples

#### Kronecker product

<!-- Lets say we have $X\in \mathbb{R}^{3 \times 3}$, $Y \in \mathbb{R}^{5 \times 5}$, and we are looking at the function

$$E : \mathbb{R}^{3 \times 3} \Rightarrow \mathbb{R}^{(3 \times 3) \times (5 \times 5)} $$
$$E(X) = X \otimes Y $$ -->

Consider $E(X) = $ `K = einsum('ij,kl->ijkl', x, y)`.

The values in `K[i, j, :, :]` are $x_{i, j} * Y$. The partial derivative of this w.r.t. $x_{i, j}$ is of course $Y$, and the partial w.r.t. $x_{m, n}$ where $(i, j) \neq (m, n)$ is 0. Now we can build our jacobian. We want a `[3, 3, 5, 5, 3, 3]` tensor where

$$
[i, j, :, :, m, n] =
\begin{cases}
Y \ \ if  \  \ (i, j) = (m, n) \\
0 \ \ \ else
\end{cases}
$$

If we use a for loop:

```python
import torch

X, Y = torch.randn(3, 3), torch.randn(5, 5)
J = torch.zeros(3, 3, 5, 5, 3, 3)

for i in range(3):
    for j in range(3):
        for k in range(5):
            for l in range(5):
                for m in range(3):
                    for n in range(3):
                        if (i == m) and (j == n):
                            J[i, j, :, :, m, n] = a
```

This works but its cringe. We can actually use some cool einsum slicing to do it in one line

```python
from torch import einsum
J = torch.zeros(3, 3, 5, 5, 3, 3)
einsum('ijklij->kl', J)[:] = Y
```

And thats the jacobian for the kronecker product. If in the original einsum we summed over a dimension, we would sum over the same dimension in the jacobian.

```python
K = einsum('ij,kl->ijk', X, Y)

J = torch.zeros(3, 3, 5, 5, 3, 3)
einsum('ijklij->kl', J)[:] = Y
J = einsum('ijklmn->ijkmn', J)
```

#### Matrix Multiplication

Let $E(X) = $ `K = einsum('ij,jk->ik', x, y)`. As we said earlier this makes a 4D tensor $K$ where

$$
K[i, j, k, l] =
\begin{cases}
x_{ij} * y_{kl} \ \ if  \  \ j = k \\
0 \ \ \ else
\end{cases}
$$

differentiating this gives us a jacobian $J$ where

$$
J[i, j, k, l, m, n] =
\begin{cases}
y_{kl} \ \ if  \  \ (i, j) = (m, n)  \ and \ j = k \\
0 \ \ \ else
\end{cases}
$$

To build this with einsum slicing and broadcasting we can do

```python
X, Y = torch.randn(3, 4), torch.randn(4, 5)
J = torch.zeros(3, 4, 4, 5, 3, 4)
einsum('ijjkij->ijk', J)[:] = Y
J = einsum('ijklmn->ilmn')
```
#### Haddamard Product 

Let $E(X) = $ `K = einsum('ij,jk->ik', x, y)`. As we said earlier this makes a 4D tensor $K$ where

$$
K[i, j, k, l] =
\begin{cases}
x_{ij} * y_{kl} \ \ if  \  \ j = k \\
0 \ \ \ else
\end{cases}
$$

differentiating this gives us a jacobian $J$ where

$$
J[i, j, k, l, m, n] =
\begin{cases}
y_{kl} \ \ if  \  \ (i, j) = (m, n)  \ and \ j = k \\
0 \ \ \ else
\end{cases}
$$

To build this with einsum slicing and broadcasting we can do

```python
X, Y = torch.randn(3, 4), torch.randn(4, 5)
J = torch.zeros(3, 4, 4, 5, 3, 4)
einsum('ijjkij->ijk', J)[:] = Y
J = einsum('ijklmn->ilmn')
```