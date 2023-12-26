import torch
import string
from torch import einsum


def parse_ptrn(ptrn):
    in_ptrn, out_ptrn = ptrn.split("->")[0].split(","), ptrn.split("->")[1]
    x_ptrn, a_ptrn = in_ptrn
    return x_ptrn, a_ptrn, out_ptrn


def flip_ptrn(ptrn):
    x_ptrn, a_ptrn, out_ptrn = parse_ptrn(ptrn)
    return f"{a_ptrn},{x_ptrn}->{out_ptrn}"


def dummy_ptrn(x_len, a_len):
    x_ptrn = string.ascii_lowercase[:x_len]
    out_ptrn = string.ascii_lowercase[x_len:a_len]
    return f"{x_ptrn},{x_ptrn + out_ptrn}->{out_ptrn}"


invert_dict = lambda x: {v: k for k, v in x.items()}


def generate_dictionaries(x_ptrn, a_ptrn):
    chars = [string.ascii_lowercase[_] for _ in range(len(x_ptrn + a_ptrn))]
    char_dict = dict(zip(chars, list(x_ptrn + a_ptrn)))
    char_dict_inv = invert_dict(char_dict)
    return char_dict, char_dict_inv


def generate_einsum_out_string(char_dict_inv, x_ptrn, a_ptrn, out_ptrn):
    start = f"{string.ascii_lowercase[:len(x_ptrn + a_ptrn + x_ptrn)]}"
    end = f"{''.join([char_dict_inv[c] for c in out_ptrn])}{start[-len(x_ptrn):]}"
    return f"{start}->{end}"


def generate_einsum_jacobian_string(x_ptrn, a_ptrn):
    start = x_ptrn + a_ptrn + x_ptrn
    end = "".join([c for c in x_ptrn if c not in a_ptrn]) + a_ptrn
    return f"{start}->{end}"


def get_strings_helper(ptrn):
    x_ptrn, a_ptrn, out_ptrn = parse_ptrn(ptrn)
    char_dict, char_dict_inv = generate_dictionaries(x_ptrn, a_ptrn)
    einsum_out_string = generate_einsum_out_string(
        char_dict_inv, x_ptrn, a_ptrn, out_ptrn
    )
    einsum_jacobian_string = generate_einsum_jacobian_string(x_ptrn, a_ptrn)
    return einsum_out_string, einsum_jacobian_string


def einsum_jacobian(ptrn, x, a):
    jacobian = torch.zeros(*x.shape, *a.shape, *x.shape)
    einsum_out_string, einsum_jacobian_string = get_strings_helper(ptrn)
    einsum(einsum_jacobian_string, jacobian)[:] = a
    return einsum(einsum_out_string, jacobian)
