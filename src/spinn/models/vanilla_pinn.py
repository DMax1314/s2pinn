from typing import Any, Dict

import torch
import torch.nn as nn


class VanillaPINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        act_fn: nn.Module
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_fn]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(-1)
        if x.ndim == 2:
            x = x.squeeze(-1)
        if y.ndim == 2:
            y = y.squeeze(-1)

        inp = torch.cat([t.unsqueeze(-1), x.unsqueeze(-1), y.unsqueeze(-1), z], dim=-1)
        return self.net(inp).squeeze(-1)

    def forward_with_derivatives(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if t.ndim == 2:
            t = t.squeeze(-1)
        if x.ndim == 2:
            x = x.squeeze(-1)
        if y.ndim == 2:
            y = y.squeeze(-1)

        t_ad = t.clone().detach().requires_grad_(True)
        x_ad = x.clone().detach().requires_grad_(True)
        y_ad = y.clone().detach().requires_grad_(True)

        inp = torch.cat(
            [t_ad.unsqueeze(-1), x_ad.unsqueeze(-1), y_ad.unsqueeze(-1), z], dim=-1
        )
        u = self.net(inp).squeeze(-1)

        ones = torch.ones_like(u)

        u_t = torch.autograd.grad(u, t_ad, ones, create_graph=True, retain_graph=True)[
            0
        ]
        du_dx = torch.autograd.grad(
            u, x_ad, ones, create_graph=True, retain_graph=True
        )[0]
        du_dy = torch.autograd.grad(
            u, y_ad, ones, create_graph=True, retain_graph=True
        )[0]

        d2u_dx2 = torch.autograd.grad(
            du_dx, x_ad, ones, create_graph=True, retain_graph=True
        )[0]
        d2u_dy2 = torch.autograd.grad(
            du_dy, y_ad, ones, create_graph=True, retain_graph=True
        )[0]

        laplacian_u = d2u_dx2 + d2u_dy2

        return {
            "u": u,
            "u_t": u_t,
            "du_dx": du_dx,
            "du_dy": du_dy,
            "laplacian_u": laplacian_u,
        }
