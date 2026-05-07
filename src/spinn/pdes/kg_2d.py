import torch
import numpy as np
from numpy import pi
from typing import Dict, Any, Tuple


def manufactured(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Manufactured solution for 2D Klein–Gordon:
    u(x, y, t) = sin(pi*x) * sin(pi*y) * cos(t)
    """
    pi = np.pi
    return torch.sin(pi * x[..., 0]) * torch.sin(pi * x[..., 1]) * torch.cos(t)


def residual(
    x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    Compute strong-form PDE residual for 2D Klein–Gordon:
    u_tt - Δu + m^2 u = f(x, t)
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_pred = u

    # First derivatives
    grads = torch.autograd.grad(
        u_pred, [x, t], grad_outputs=torch.ones_like(u_pred), create_graph=True
    )
    du_dx = grads[0][..., 0]
    du_dy = grads[0][..., 1]
    du_dt = grads[1]

    # Second derivatives
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0][..., 0]
    d2u_dy2 = torch.autograd.grad(
        du_dy, x, grad_outputs=torch.ones_like(du_dy), create_graph=True
    )[0][..., 1]
    d2u_tt = torch.autograd.grad(
        du_dt, t, grad_outputs=torch.ones_like(du_dt), create_graph=True
    )[0]

    laplacian = d2u_dx2 + d2u_dy2
    m = params.get("m", 1.0)
    f = manufactured(x, t) * (
        -(pi**2) * 2 * torch.cos(t)
        - m**2 * torch.sin(pi * x[..., 0]) * torch.sin(pi * x[..., 1]) * torch.cos(t)
    )
    res = d2u_tt - laplacian + m**2 * u_pred - f
    return res


def domain_sampler(
    Nc: int, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample Nc points in [0,1]^2 x [0,1] (space-time)
    Returns x: [Nc,2], t: [Nc,1]
    """
    x = torch.rand(Nc, 2, device=device)
    t = torch.rand(Nc, 1, device=device)
    return x, t
