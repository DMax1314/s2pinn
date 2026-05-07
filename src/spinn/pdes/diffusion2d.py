import torch
import numpy as np
from typing import Dict, Any, Tuple

def manufactured(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Manufactured solution for 2D nonlinear diffusion:
    u(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-t)
    """
    pi = np.pi
    return torch.sin(pi * x) * torch.sin(pi * x[..., 1]) * torch.exp(-t)

def residual(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    """
    Compute strong-form PDE residual for 2D nonlinear diffusion:
    u_t = div(D(u) * grad u)
    D(u) = 1 + u^2
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_pred = u

    grads = torch.autograd.grad(u_pred, [x, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)
    du_dx = grads[0][..., 0]
    du_dy = grads[0][..., 1]
    du_dt = grads[1]

    grad_u = torch.stack([du_dx, du_dy], dim=-1)
    D = 1.0 + u_pred ** 2

    # Compute divergence: div(D * grad u)
    D_grad_u = D.unsqueeze(-1) * grad_u
    div = torch.zeros_like(u_pred)
    for i in range(2):
        div += torch.autograd.grad(D_grad_u[..., i], x, grad_outputs=torch.ones_like(D_grad_u[..., i]), create_graph=True)[0][..., i]

    res = du_dt - div
    return res

def domain_sampler(Nc: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample Nc points in [0,1]^2 x [0,1] (space-time)
    Returns x: [Nc,2], t: [Nc,1]
    """
    x = torch.rand(Nc, 2, device=device)
    t = torch.rand(Nc, 1, device=device)
    return x, t

def boundary_sampler(Nb: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Nb boundary points (x or y = 0/1)
    """
    x = torch.rand(Nb, 2, device=device)
    t = torch.rand(Nb, 1, device=device)
    # Randomly set one coordinate to 0 or 1
    idx = torch.randint(0, 2, (Nb,))
    val = torch.randint(0, 2, (Nb,)).float()
    for i in range(Nb):
        x[i, idx[i]] = val[i]
    return x, t

def initial_sampler(Ni: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Ni initial condition points (t=0)
    """
    x = torch.rand(Ni, 2, device=device)
    t = torch.zeros(Ni, 1, device=device)
    return x, t
