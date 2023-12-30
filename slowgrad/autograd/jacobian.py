import torch
import string
from torch import einsum


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


def create_backprop_einsum_pattern(x_len: int, a_len: int) -> str:
    """Constructs an einsum pattern for backpropagation."""
    x_ptrn = string.ascii_lowercase[:x_len]
    out_ptrn = string.ascii_lowercase[x_len:a_len]
    return f"{x_ptrn},{x_ptrn + out_ptrn}->{out_ptrn}"


def create_output_einsum_pattern(
    char_dict_inv: dict, x_ptrn: str, a_ptrn: str, out_ptrn: str
) -> str:
    """Creates einsum pattern to ensure the outputed Jacobian is the right shape"""
    current_ptrn = f"{string.ascii_lowercase[:len(x_ptrn + a_ptrn + x_ptrn)]}"
    final_ptrn = (
        f"{''.join([char_dict_inv[c] for c in out_ptrn])}{current_ptrn[-len(x_ptrn):]}"
    )
    return f"{current_ptrn}->{final_ptrn}"


def create_jacobian_einsum_pattern(x_ptrn: str, a_ptrn: str) -> str:
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
    einsum_jacobian_string = create_jacobian_einsum_pattern(x_ptrn, a_ptrn)
    return einsum_out_string, einsum_jacobian_string


def compute_einsum_jacobian(
    ptrn: str, x: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """Computes the Jacobian using einsum with provided tensors and pattern."""
    jacobian = torch.zeros(*x.shape, *a.shape, *x.shape)
    einsum_out_string, einsum_jacobian_string = prepare_einsum_patterns(ptrn)
    einsum(einsum_jacobian_string, jacobian)[:] = a
    return einsum(einsum_out_string, jacobian)


def sigmoid_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Computs the jacobian of sigmoid(x) for arbitary x"""
    sig = torch.sigmoid(x)
    j = torch.zeros(*x.shape, *x.shape)
    diag_ptrn = string.ascii_lowercase[: x.dim()]
    einsum(f"{diag_ptrn}{diag_ptrn}->{diag_ptrn}", j)[:] = sig * (1 - sig)
    return j


def mse_jacobian(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "shape mismatch"
    return 2 * (x - y) / x.numel()
