from typing import Any

import torch
import torch.nn as nn

from spinn.models.bases import GPCBasis, SpatialDict2D, TimeBasis


class CPCore(nn.Module):
    def __init__(self, M: int, Mt: int, G: int, rank: int):
        super().__init__()
        if M <= 0:
            raise ValueError(f"M must be positive, got {M}")
        if Mt <= 0:
            raise ValueError(f"Mt must be positive, got {Mt}")
        if G <= 0:
            raise ValueError(f"G must be positive, got {G}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.M: int = M
        self.Mt: int = Mt
        self.G: int = G
        self.rank: int = rank

        self.A: nn.Parameter = nn.Parameter(torch.empty(M, rank))
        self.B: nn.Parameter = nn.Parameter(torch.empty(Mt, rank))
        self.D: nn.Parameter = nn.Parameter(torch.empty(G, rank))
        self.w: nn.Parameter = nn.Parameter(torch.ones(rank))

        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)
        nn.init.orthogonal_(self.D)

    def forward(
        self,
        F_space: torch.Tensor,
        F_time: torch.Tensor,
        F_stoc: torch.Tensor,
    ) -> torch.Tensor:
        if F_space.ndim != 2 or F_space.shape[1] != self.M:
            raise ValueError(
                f"F_space must have shape [B, {self.M}], got {tuple(F_space.shape)}"
            )
        if F_time.ndim != 2 or F_time.shape[1] != self.Mt:
            raise ValueError(
                f"F_time must have shape [B, {self.Mt}], got {tuple(F_time.shape)}"
            )
        if F_stoc.ndim != 2 or F_stoc.shape[1] != self.G:
            raise ValueError(
                f"F_stoc must have shape [B, {self.G}], got {tuple(F_stoc.shape)}"
            )

        batch_size = F_space.shape[0]
        if F_time.shape[0] != batch_size or F_stoc.shape[0] != batch_size:
            raise ValueError(
                "F_space, F_time, and F_stoc must share the same batch dimension, got "
                f"{batch_size}, {F_time.shape[0]}, {F_stoc.shape[0]}"
            )

        sa = F_space @ self.A
        tb = F_time @ self.B
        sd = F_stoc @ self.D
        u = (sa * tb * sd) @ self.w

        assert u.shape == (batch_size,)
        return u

    def normalize_factors(self) -> None:
        with torch.no_grad():
            for factor in (self.A, self.B, self.D):
                col_norms = torch.linalg.norm(factor, dim=0).clamp_min(1e-12)
                factor.div_(col_norms.unsqueeze(0))
                self.w.mul_(col_norms)

    def orthogonality_reg(self) -> torch.Tensor:
        reg = self.A.new_tensor(0.0)
        for factor in (self.A, self.B, self.D):
            col_norms = torch.linalg.norm(factor, dim=0, keepdim=True).clamp_min(1e-12)
            factor_norm = factor / col_norms
            gram = factor_norm.transpose(0, 1) @ factor_norm
            identity = torch.eye(self.rank, dtype=factor.dtype, device=factor.device)
            reg = reg + ((gram - identity) ** 2).mean()
        return reg


class ProductModelGPC(nn.Module):
    def __init__(
        self,
        spatial_cfg: dict[str, Any],
        temporal_cfg: dict[str, Any],
        stochastic_cfg: dict[str, Any],
        cp_cfg: dict[str, Any],
    ):
        super().__init__()
        if "rank" not in cp_cfg:
            raise ValueError("cp_cfg must contain key 'rank'")

        self.spatial = SpatialDict2D(**spatial_cfg)
        self.temporal = TimeBasis(**temporal_cfg)
        self.stochastic = GPCBasis(**stochastic_cfg)

        M = self.spatial.n_atoms
        Mt = self.temporal.num_features
        G = self.stochastic.num_basis
        self.cp = CPCore(M=M, Mt=Mt, G=G, rank=int(cp_cfg["rank"]))

    @staticmethod
    def _as_vector(coord: torch.Tensor, name: str) -> torch.Tensor:
        if coord.ndim == 1:
            return coord
        if coord.ndim == 2 and coord.shape[1] == 1:
            return coord[:, 0]
        raise ValueError(
            f"{name} must have shape [B] or [B, 1], got {tuple(coord.shape)}"
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")

        F_space = self.spatial(x_vec, y_vec)
        F_time, _ = self.temporal(t)
        F_stoc = self.stochastic(z)
        u = self.cp(F_space, F_time, F_stoc)
        return u

    def forward_with_derivatives(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")

        if not x_vec.requires_grad or not x_vec.is_leaf:
            x_vec = x_vec.clone().detach().requires_grad_(True)
        if not y_vec.requires_grad or not y_vec.is_leaf:
            y_vec = y_vec.clone().detach().requires_grad_(True)

        F_space = self.spatial(x_vec, y_vec)
        F_time, dF_time_dt = self.temporal(t)
        F_stoc = self.stochastic(z)

        u = self.cp(F_space, F_time, F_stoc)
        u_t = self.cp(F_space, dF_time_dt, F_stoc)

        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x_vec,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        du_dy = torch.autograd.grad(
            outputs=u,
            inputs=y_vec,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=x_vec,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0]
        d2u_dy2 = torch.autograd.grad(
            outputs=du_dy,
            inputs=y_vec,
            grad_outputs=torch.ones_like(du_dy),
            create_graph=True,
            retain_graph=True,
        )[0]

        laplacian_u = d2u_dx2 + d2u_dy2

        return {
            "u": u,
            "u_t": u_t,
            "du_dx": du_dx,
            "du_dy": du_dy,
            "laplacian_u": laplacian_u,
        }

    def gram_penalty(self, F_space: torch.Tensor) -> torch.Tensor:
        if F_space.ndim != 2:
            raise ValueError(
                f"F_space must have shape [B, M], got {tuple(F_space.shape)}"
            )

        batch_size, num_atoms = F_space.shape
        if batch_size <= 0:
            raise ValueError("F_space batch size must be positive")

        gram_hat = F_space.transpose(0, 1) @ F_space / float(batch_size)
        identity = torch.eye(num_atoms, device=F_space.device, dtype=F_space.dtype)
        return ((gram_hat - identity) ** 2).mean()

    def regularization(
        self,
        F_space: torch.Tensor,
        w_orth: float,
        w_fac_orth: float,
    ) -> torch.Tensor:
        return (
            w_orth * self.gram_penalty(F_space)
            + w_fac_orth * self.cp.orthogonality_reg()
        )
