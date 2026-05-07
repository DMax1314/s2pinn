from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _mlp(in_dim: int, width: int, depth: int, activation: str) -> nn.Sequential:
    if in_dim <= 0:
        raise ValueError(f"in_dim must be positive, got {in_dim}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    act = _activation(activation)
    layers: list[nn.Module] = [nn.Linear(in_dim, width), act]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), act])
    return nn.Sequential(*layers)


class PIDeepONet(nn.Module):
    def __init__(
        self,
        R: int,
        latent_dim: int = 128,
        branch_width: int = 128,
        branch_depth: int = 3,
        trunk_width: int = 128,
        trunk_depth: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.R = int(R)
        self.latent_dim = int(latent_dim)

        self.branch = _mlp(R, branch_width, branch_depth, activation)
        self.trunk = _mlp(3, trunk_width, trunk_depth, activation)
        self.branch_out = nn.Linear(branch_width, latent_dim)
        self.trunk_out = nn.Linear(trunk_width, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _as_vector(coord: torch.Tensor, name: str) -> torch.Tensor:
        if coord.ndim == 1:
            return coord
        if coord.ndim == 2 and coord.shape[1] == 1:
            return coord[:, 0]
        raise ValueError(
            f"{name} must have shape [B] or [B,1], got {tuple(coord.shape)}"
        )

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        t_v = self._as_vector(t, "t")
        x_v = self._as_vector(x, "x")
        y_v = self._as_vector(y, "y")
        if Z.ndim != 2 or Z.shape[1] != self.R:
            raise ValueError(f"Z must have shape [B, {self.R}], got {tuple(Z.shape)}")

        inp_trunk = torch.stack([t_v, x_v, y_v], dim=-1)
        trunk_lat = self.trunk_out(self.trunk(inp_trunk))
        branch_lat = self.branch_out(self.branch(Z))
        u = (trunk_lat * branch_lat).sum(dim=-1) + self.bias
        return u

    def forward_with_derivatives(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t_v = self._as_vector(t, "t")
        x_v = self._as_vector(x, "x")
        y_v = self._as_vector(y, "y")

        if not t_v.requires_grad or not t_v.is_leaf:
            t_v = t_v.clone().detach().requires_grad_(True)
        if not x_v.requires_grad or not x_v.is_leaf:
            x_v = x_v.clone().detach().requires_grad_(True)
        if not y_v.requires_grad or not y_v.is_leaf:
            y_v = y_v.clone().detach().requires_grad_(True)

        u = self.forward(t_v, x_v, y_v, Z)
        ones = torch.ones_like(u)

        u_t = torch.autograd.grad(u, t_v, ones, create_graph=True, retain_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_v, ones, create_graph=True, retain_graph=True)[
            0
        ]
        du_dy = torch.autograd.grad(u, y_v, ones, create_graph=True, retain_graph=True)[
            0
        ]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_v, ones, create_graph=True, retain_graph=True
        )[0]
        d2u_dy2 = torch.autograd.grad(
            du_dy, y_v, ones, create_graph=True, retain_graph=True
        )[0]
        laplacian_u = d2u_dx2 + d2u_dy2

        return {
            "u": u,
            "u_t": u_t,
            "du_dx": du_dx,
            "du_dy": du_dy,
            "laplacian_u": laplacian_u,
        }
