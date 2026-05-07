import math
from itertools import product
from typing import Literal, Optional, cast

import torch
import torch.nn as nn


class GPCBasis(nn.Module):
    """Tensorized gPC basis for R random variables, total degree p."""

    def __init__(self, R: int, p: int, dist: str = "gaussian"):
        super().__init__()
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if p < 0:
            raise ValueError(f"p must be non-negative, got {p}")
        if dist not in {"gaussian", "uniform"}:
            raise ValueError(f"dist must be 'gaussian' or 'uniform', got {dist}")

        self.R: int = R
        self.p: int = p
        self.dist: Literal["gaussian", "uniform"] = cast(
            Literal["gaussian", "uniform"], dist
        )

        indices: list[tuple[int, ...]] = [
            idx for idx in product(range(p + 1), repeat=R) if sum(idx) <= p
        ]
        expected_num_basis = math.comb(R + p, p)
        if len(indices) != expected_num_basis:
            raise RuntimeError(
                f"Invalid multi-index count: got {len(indices)}, expected {expected_num_basis}"
            )

        multi_indices = torch.tensor(indices, dtype=torch.long)
        self.multi_indices: torch.Tensor
        self.register_buffer("multi_indices", multi_indices, persistent=True)

    @property
    def num_basis(self) -> int:
        return int(self.multi_indices.shape[0])

    def _eval_hermite(self, x: torch.Tensor, max_degree: int) -> torch.Tensor:
        values = [torch.ones_like(x)]
        if max_degree == 0:
            return torch.stack(values, dim=-1)

        values.append(x)
        for n in range(1, max_degree):
            values.append(x * values[n] - float(n) * values[n - 1])
        return torch.stack(values, dim=-1)

    def _eval_legendre(self, x: torch.Tensor, max_degree: int) -> torch.Tensor:
        values = [torch.ones_like(x)]
        if max_degree == 0:
            return torch.stack(values, dim=-1)

        values.append(x)
        for n in range(1, max_degree):
            numerator = (2.0 * float(n) + 1.0) * x * values[n] - float(n) * values[
                n - 1
            ]
            values.append(numerator / float(n + 1))
        return torch.stack(values, dim=-1)

    def _eval_univariate(self, x: torch.Tensor, max_degree: int) -> torch.Tensor:
        if self.dist == "gaussian":
            return self._eval_hermite(x, max_degree)
        return self._eval_legendre(x, max_degree)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, R] random variable samples

        Returns:
            [B, G] gPC basis evaluations
        """
        if z.ndim != 2 or z.shape[1] != self.R:
            raise ValueError(f"z must have shape [B, {self.R}], got {tuple(z.shape)}")

        batch_size = z.shape[0]
        features = z.new_ones((batch_size, self.num_basis))
        max_degree = self.p

        for r in range(self.R):
            univariate = self._eval_univariate(z[:, r], max_degree)
            degrees_r = self.multi_indices[:, r]
            features = features * univariate.index_select(1, degrees_r)

        assert features.shape == (batch_size, self.num_basis)
        return features


class TimeBasis(nn.Module):
    """DC + sine/cosine with learnable frequency scales."""

    def __init__(self, n_freq: int = 8):
        super().__init__()
        if n_freq < 0:
            raise ValueError(f"n_freq must be non-negative, got {n_freq}")

        self.n_freq: int = n_freq
        if n_freq == 0:
            init = torch.empty(0)
        else:
            max_log_omega = math.log(2.0 * math.pi * float(n_freq))
            init = torch.empty(n_freq).uniform_(0.0, max_log_omega)
        self.log_omega: nn.Parameter = nn.Parameter(init)

    @property
    def num_features(self) -> int:
        return 1 + 2 * self.n_freq

    @staticmethod
    def _as_vector(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t
        if t.ndim == 2 and t.shape[1] == 1:
            return t[:, 0]
        raise ValueError(f"t must have shape [B] or [B, 1], got {tuple(t.shape)}")

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t: [B] or [B, 1] time values

        Returns:
            (F_time, dF_time_dt)
                F_time: [B, M_t]
                dF_time_dt: [B, M_t]
        """
        t_vec = self._as_vector(t)
        batch_size = t_vec.shape[0]

        dc = torch.ones(batch_size, 1, dtype=t_vec.dtype, device=t_vec.device)
        d_dc = torch.zeros(batch_size, 1, dtype=t_vec.dtype, device=t_vec.device)

        if self.n_freq == 0:
            assert dc.shape == (batch_size, self.num_features)
            assert d_dc.shape == (batch_size, self.num_features)
            return dc, d_dc

        omega = torch.exp(self.log_omega).to(device=t_vec.device, dtype=t_vec.dtype)
        phase = t_vec.unsqueeze(1) * omega.unsqueeze(0)
        cos_term = torch.cos(phase)
        sin_term = torch.sin(phase)

        features = torch.cat([dc, cos_term, sin_term], dim=1)
        d_cos = -omega.unsqueeze(0) * sin_term
        d_sin = omega.unsqueeze(0) * cos_term
        derivatives = torch.cat([d_dc, d_cos, d_sin], dim=1)

        assert features.shape == (batch_size, self.num_features)
        assert derivatives.shape == (batch_size, self.num_features)
        return features, derivatives


class SpatialDict2D(nn.Module):
    """Anisotropic Gaussian dictionary with optional residual coordinate warp."""

    def __init__(
        self,
        n_atoms: int = 256,
        init_sigma: float = 0.15,
        use_warp: bool = True,
        warp_hidden: int = 64,
        warp_eps: float = 0.1,
    ):
        super().__init__()
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if init_sigma <= 0.0:
            raise ValueError(f"init_sigma must be positive, got {init_sigma}")
        if warp_hidden <= 0:
            raise ValueError(f"warp_hidden must be positive, got {warp_hidden}")

        self.n_atoms: int = n_atoms
        self.use_warp: bool = use_warp
        self.warp_eps: float = warp_eps

        centers = torch.empty(n_atoms, 2).uniform_(-1.0, 1.0)
        self.centers: nn.Parameter = nn.Parameter(centers)

        init_log_sigma = torch.full((n_atoms,), math.log(init_sigma))
        self.log_sigma: nn.Parameter = nn.Parameter(init_log_sigma)

        self.warp_mlp: Optional[nn.Module]
        if use_warp:
            self.warp_mlp = nn.Sequential(
                nn.Linear(2, warp_hidden),
                nn.Tanh(),
                nn.Linear(warp_hidden, 2),
            )
        else:
            self.warp_mlp = None

    @staticmethod
    def _as_vector(coord: torch.Tensor, name: str) -> torch.Tensor:
        if coord.ndim == 1:
            return coord
        if coord.ndim == 2 and coord.shape[1] == 1:
            return coord[:, 0]
        raise ValueError(
            f"{name} must have shape [B] or [B, 1], got {tuple(coord.shape)}"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B], y: [B] spatial coordinates

        Returns:
            F_space: [B, M] spatial features
        """
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        if x_vec.shape != y_vec.shape:
            raise ValueError(
                f"x and y must match shape, got {tuple(x_vec.shape)} and {tuple(y_vec.shape)}"
            )

        xy = torch.stack([x_vec, y_vec], dim=-1)
        batch_size = xy.shape[0]
        assert xy.shape == (batch_size, 2)

        if self.use_warp:
            if self.warp_mlp is None:
                raise RuntimeError("use_warp=True but warp_mlp is None")
            warp_delta = cast(torch.Tensor, self.warp_mlp(xy))
            xy_warped = xy + self.warp_eps * warp_delta
        else:
            xy_warped = xy

        sigma = torch.exp(self.log_sigma).clamp_min(1e-8)
        diff = xy_warped.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        denom = 2.0 * (sigma * sigma).unsqueeze(0)
        features = torch.exp(-dist_sq / denom)

        assert features.shape == (batch_size, self.n_atoms)
        return features
