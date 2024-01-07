import torch
import string
from torch import einsum
import torch.nn.functional as F
from typing import Optional


def split_einsum_pattern(ptrn: str) -> tuple[str, str, str]:
    """Splits einsum pattern into input and output components."""
    in_ptrn, out_ptrn = ptrn.split("->")[0].split(","), ptrn.split("->")[1]
    x_ptrn, a_ptrn = in_ptrn
    return x_ptrn, a_ptrn, out_ptrn


def swap_einsum_inputs(ptrn: str) -> str:
    """Swaps the first two input patterns of an einsum equation."""
    x_ptrn, a_ptrn, out_ptrn = split_einsum_pattern(ptrn)
    return f"{a_ptrn},{x_ptrn}->{out_ptrn}"


def invert_dict(x: dict) -> dict:
    """Inverts a dict"""
    return {v: k for k, v in x.items()}


def create_char_mappings(x_ptrn: str, a_ptrn: str) -> tuple[dict, dict]:
    """Generates forward and inverse character mappings for einsum patterns."""
    chars = [string.ascii_lowercase[_] for _ in range(len(x_ptrn + a_ptrn))]
    char_dict = dict(zip(chars, list(x_ptrn + a_ptrn)))
    char_dict_inv = invert_dict(char_dict)
    return char_dict, char_dict_inv


def create_backprop_einsum_pattern(upstream_dim: int, local_dim: int) -> str:
    """Constructs an einsum pattern for backpropagation."""
    upstream_ptrn = string.ascii_lowercase[:upstream_dim]
    local_ptrn = string.ascii_lowercase[upstream_dim:local_dim]
    return f"{upstream_ptrn},{upstream_ptrn + local_ptrn}->{local_ptrn}"


def backprop(
    upstream_jacobian: torch.Tensor, local_jacobian: torch.Tensor
) -> torch.Tensor:
    """Backpropogates upstream jacobian into local jacobian"""
    ptrn = create_backprop_einsum_pattern(upstream_jacobian.dim(), local_jacobian.dim())
    return einsum(ptrn, upstream_jacobian, local_jacobian)


def create_output_einsum_pattern(
    char_dict_inv: dict, x_ptrn: str, a_ptrn: str, out_ptrn: str
) -> str:
    """Creates einsum pattern to ensure the outputed Jacobian is the right shape"""
    current_ptrn = f"{string.ascii_lowercase[:len(x_ptrn + a_ptrn + x_ptrn)]}"
    final_ptrn = (
        f"{''.join([char_dict_inv[c] for c in out_ptrn])}{current_ptrn[-len(x_ptrn):]}"
    )
    return f"{current_ptrn}->{final_ptrn}"


def create_jacobian_diagonal_pattern(x_ptrn: str, a_ptrn: str) -> str:
    """Creates einsum pattern to select a slice of the Jacobian"""
    start = x_ptrn + a_ptrn + x_ptrn
    end = "".join([c for c in x_ptrn if c not in a_ptrn]) + a_ptrn
    return f"{start}->{end}"


def prepare_einsum_patterns(ptrn: str) -> tuple[str, str]:
    """Prepares einsum patterns for Jacobian calculation."""
    x_ptrn, a_ptrn, out_ptrn = split_einsum_pattern(ptrn)
    char_dict, char_dict_inv = create_char_mappings(x_ptrn, a_ptrn)
    einsum_out_string = create_output_einsum_pattern(
        char_dict_inv, x_ptrn, a_ptrn, out_ptrn
    )
    einsum_jacobian_string = create_jacobian_diagonal_pattern(x_ptrn, a_ptrn)
    return einsum_out_string, einsum_jacobian_string


def compute_einsum_jacobian(
    ptrn: str, x: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """Computes the jacobian of einsum(ptrn, x, a) w.r.t. x"""
    jacobian = torch.zeros(*x.shape, *a.shape, *x.shape)
    einsum_out_string, einsum_jacobian_string = prepare_einsum_patterns(ptrn)
    einsum(einsum_jacobian_string, jacobian)[:] = a
    return einsum(einsum_out_string, jacobian)


def diag_ptrn(dim: int) -> str:
    """Returns a einsum ptrn used to select the diagonal of a tensor"""
    ptrn = string.ascii_lowercase[:dim]
    return f"{ptrn}{ptrn}->{ptrn}"


def sigmoid_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Computes the jacobian of sigmoid(x) for arbitary x"""
    sig = torch.sigmoid(x)
    j = torch.zeros(*x.shape, *x.shape)
    einsum(diag_ptrn(x.dim()), j)[:] = sig * (1 - sig)
    return j


def relu_jacobian(x: torch.Tensor) -> torch.Tensor:
    j = torch.zeros(*x.shape, *x.shape)
    diag = torch.where(x > 0, 1, 0)
    einsum(diag_ptrn(x.dim()), j)[:] = diag
    return j


def mse_jacobian(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes jacobian of MSE loss w.r.t x"""
    return 2 * (x - y) / x.numel()


def softmax_jacobian(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes the jacobian of the softmax of x"""
    identity = torch.zeros(*x.shape, *x.shape)
    einsum(diag_ptrn(x.dim()), identity)[:] = 1
    soft = F.softmax(x, dim=dim)

    chars = string.ascii_lowercase[: 2 * x.dim()]

    def make_mask(x, dim=dim):
        masks = [torch.eye(s) for s in x.shape]
        masks[dim] = masks[dim].fill_(1)
        in_ptrn = ",".join(chars[i : i + 2] for i in range(0, len(chars), 2))
        out_ptrn = chars[::2] + chars[1::2]
        return einsum(f"{in_ptrn}->{out_ptrn}", *masks)

    mask = make_mask(x, dim=dim)

    return einsum(
        f"{chars[:x.dim()]},{chars},{chars}->{chars}", soft, identity - soft, mask
    )


def haddamard_sum_ptrn(x_dim: int) -> str:
    chars = string.ascii_lowercase[:x_dim]
    return f"{chars},{chars}->"


def cross_entropy_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    soft_x: Optional[torch.Tensor] = None,
    log_soft_x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes jacobian of cross entropy loss w.r.t. x"""
    dim = -1
    if soft_x is not None and log_soft_x is not None:
        soft, log_soft = soft_x, log_soft_x
    else:
        soft = F.softmax(x, dim=dim)
        log_soft = torch.log(soft)

    haddamard_jacobian = -(1 / x.shape[0]) * compute_einsum_jacobian(
        haddamard_sum_ptrn(x.dim()), log_soft, y
    )
    log_jacobian = torch.zeros(*x.shape, *x.shape)
    einsum(diag_ptrn(x.dim()), log_jacobian)[:] = 1 / soft
    soft_jacobian = softmax_jacobian(x, dim=dim)
    j = backprop(haddamard_jacobian, log_jacobian)
    return backprop(j, soft_jacobian)
