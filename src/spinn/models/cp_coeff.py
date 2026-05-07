import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

class CPCoeff(nn.Module):
    """
    CP decomposition coefficient tensor: a⊗b⊗g
    Shapes:
        - factors: List of [dim_i, rank] for each mode
        - output: [dim_1, dim_2, ..., dim_N] via CP
    """
    def __init__(self, dims, rank: int, init: str = "normal"):
        super().__init__()
        self.dims = dims
        self.rank = rank
        self.factors = nn.ParameterList([
            nn.Parameter(self._init_factor(d, rank, init)) for d in dims
        ])

    def _init_factor(self, d, r, init):
        if init == "normal":
            return torch.randn(d, r) * 0.1
        elif init == "uniform":
            return torch.rand(d, r) * 0.2 - 0.1
        else:
            raise ValueError(f"Unknown init: {init}")

    def forward(self):
        # Returns CP tensor as sum_k a[:,k]⊗b[:,k]⊗g[:,k]
        # For N modes: einsum over rank
        out = self.factors[0]
        for f in self.factors[1:]:
            out = torch.einsum("ir,jr->ijr", out, f).reshape(-1, self.rank)
        # Sum over rank to get full tensor
        return out.sum(-1).reshape(*self.dims)

    def cp_tensor(self):
        # Returns full tensor (not memory efficient for large dims)
        return self.forward()

    def orthogonality_reg(self, weight: float = 1.0):
        # Sum of squared off-diagonal elements of Gram matrices
        reg = 0.0
        for f in self.factors:
            gram = f.t() @ f  # [rank, rank]
            off_diag = gram - torch.diag(torch.diag(gram))
            reg += weight * (off_diag ** 2).sum()
        return reg

class FullCoeff(nn.Module):
    """
    Full coefficient tensor (no decomposition)
    Shape: dims = [d1, d2, ..., dN]
    """
    def __init__(self, dims, init: str = "normal"):
        super().__init__()
        self.dims = dims
        numel = 1
        for d in dims:
            numel *= d
        if init == "normal":
            arr = torch.randn(numel) * 0.1
        elif init == "uniform":
            arr = torch.rand(numel) * 0.2 - 0.1
        else:
            raise ValueError(f"Unknown init: {init}")
        self.coeff = nn.Parameter(arr.reshape(*dims))

    def forward(self):
        return self.coeff

    def gram_reg(self, weight: float = 1.0, mode: int = 0):
        # Gram regularization on mode-n unfolding
        unfold = self.coeff.reshape(self.dims[mode], -1)
        gram = unfold @ unfold.t()
        off_diag = gram - torch.diag(torch.diag(gram))
        return weight * (off_diag ** 2).sum()

def get_coeff_module(
    coeff_type: Literal["cp", "full"],
    dims,
    rank: Optional[int] = None,
    init: str = "normal"
):
    if coeff_type == "cp":
        if rank is None:
            raise ValueError("CP requires rank")
        return CPCoeff(dims, rank, init)
    elif coeff_type == "full":
        return FullCoeff(dims, init)
    else:
        raise ValueError(f"Unknown coeff_type: {coeff_type}")
