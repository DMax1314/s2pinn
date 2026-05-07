import torch
import numpy as np
from numpy import pi
from typing import Dict, Any, Tuple


def manufactured(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Manufactured solution for 3D Navier–Stokes (MMS):
    u(x, y, z, t) = sin(pi*x) * sin(pi*y) * sin(pi*z) * exp(-t)
    """
    pi = np.pi
    return (
        torch.sin(pi * x[..., 0])
        * torch.sin(pi * x[..., 1])
        * torch.sin(pi * x[..., 2])
        * torch.exp(-t)
    )


def residual(
    x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    Compute strong-form PDE residual for 3D Navier–Stokes (MMS, vorticity form):
    u_t + (u · ∇)u = -∇p + ν Δu + f(x, t)
    For manufactured solution, f(x, t) is chosen so u satisfies the equation.
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_pred = u

    grads = torch.autograd.grad(
        u_pred, [x, t], grad_outputs=torch.ones_like(u_pred), create_graph=True
    )
    du_dx = grads[0][..., 0]
    du_dy = grads[0][..., 1]
    du_dz = grads[0][..., 2]
    du_dt = grads[1]

    # Second derivatives
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0][..., 0]
    d2u_dy2 = torch.autograd.grad(
        du_dy, x, grad_outputs=torch.ones_like(du_dy), create_graph=True
    )[0][..., 1]
    d2u_dz2 = torch.autograd.grad(
        du_dz, x, grad_outputs=torch.ones_like(du_dz), create_graph=True
    )[0][..., 2]

    laplacian = d2u_dx2 + d2u_dy2 + d2u_dz2
    nu = params.get("nu", 0.01)
    # Nonlinear term (u · ∇)u
    grad_u = torch.stack([du_dx, du_dy, du_dz], dim=-1)
    nonlinear = (u_pred.unsqueeze(-1) * grad_u).sum(-1)
    # For MMS, f(x, t) is chosen so residual is zero
    f = manufactured(x, t) * (
        pi**2 * 3 * torch.exp(-t) - nu * pi**2 * 3 * torch.exp(-t)
    )
    res = du_dt + nonlinear - nu * laplacian - f
    return res


def domain_sampler(
    Nc: int, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample Nc points in [0,1]^3 x [0,1] (space-time)
    Returns x: [Nc,3], t: [Nc,1]
    """
    x = torch.rand(Nc, 3, device=device)
    t = torch.rand(Nc, 1, device=device)
    return x, t
