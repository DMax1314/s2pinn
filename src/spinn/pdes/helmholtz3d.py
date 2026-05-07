import torch
import numpy as np
from numpy import pi
from typing import Dict, Any, Tuple


def manufactured(x: torch.Tensor) -> torch.Tensor:
    """
    Manufactured solution for 3D Helmholtz:
    u(x, y, z) = sin(pi*x) * sin(pi*y) * sin(pi*z)
    """
    pi = np.pi
    return (
        torch.sin(pi * x[..., 0])
        * torch.sin(pi * x[..., 1])
        * torch.sin(pi * x[..., 2])
    )


def residual(x: torch.Tensor, u: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    """
    Compute strong-form PDE residual for 3D Helmholtz:
    -Δu - k^2 u = f(x)
    """
    x.requires_grad_(True)
    u_pred = u

    grads = torch.autograd.grad(
        u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True
    )[0]
    du_dx = grads[..., 0]
    du_dy = grads[..., 1]
    du_dz = grads[..., 2]

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
    k = params.get("k", pi)  # Default k=pi
    f = manufactured(x) * (3 * pi**2 - k**2)
    res = -laplacian - k**2 * u_pred - f
    return res


def domain_sampler(Nc: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Uniformly sample Nc points in [0,1]^3
    Returns x: [Nc,3]
    """
    x = torch.rand(Nc, 3, device=device)
    return x
