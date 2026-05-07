"""Stochastic Darcy flow on 2D spatial domain with NO exact solution.

PDE:  -div(k(x,y,ω) ∇u) = f    on  Ω
      u = 0                    on ∂Ω

Diffusion coefficient:
    k(x,y,Z) = exp(μ_k + σ Σᵢ √λᵢ Zᵢ φᵢ(x,y))

Source term:
    f(x,y) = 1.0

This PDE class does NOT have an exact_solution method. It is used for training
against the PDE residual and boundary conditions, then evaluated against an
external high-fidelity reference solution (e.g. from FDM Monte Carlo).
"""

import math
from typing import Dict, Tuple, Union

import torch


class StochasticDarcyMC2D:
    def __init__(
        self,
        R: int = 3,
        sigma: float = 1.0,
        corr_length: float = 1.0,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),  # Not actively used (steady state)
        mu_k: float = 0.0,
        z_distribution: str = "gaussian",
        f_value: float = 1.0,
    ):
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if corr_length <= 0.0:
            raise ValueError(f"corr_length must be positive, got {corr_length}")

        self.R = int(R)
        self.sigma = float(sigma)
        self.corr_length = float(corr_length)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.t_range = (float(t_range[0]), float(t_range[1]))
        self.mu_k = float(mu_k)
        self.z_distribution = z_distribution
        self.f_value = f_value

        kl_eigenvalues, kl_mode_x, kl_mode_y = self._compute_kl_modes()
        self.kl_eigenvalues = kl_eigenvalues
        self.kl_mode_x = kl_mode_x
        self.kl_mode_y = kl_mode_y

    @staticmethod
    def _as_vector(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == 1:
            return value
        if value.ndim == 2 and value.shape[1] == 1:
            return value[:, 0]
        raise ValueError(f"{name} must have shape [B] or [B, 1]")

    @staticmethod
    def _batch_uniform(
        n: int, bounds: Tuple[float, float], device, dtype
    ) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1, device=device, dtype=dtype)

    def _compute_kl_modes(self):
        one_d_modes = int(math.ceil(math.sqrt(float(self.R)))) + 4
        mode_ids = torch.arange(1, one_d_modes + 1, dtype=torch.long)
        mode_ids_float = mode_ids.to(dtype=torch.float64)

        x_length = self.x_range[1] - self.x_range[0]
        y_length = self.y_range[1] - self.y_range[0]
        scale_x = mode_ids_float * torch.pi * self.corr_length / x_length
        scale_y = mode_ids_float * torch.pi * self.corr_length / y_length

        lam_x = 2.0 * self.corr_length / (1.0 + scale_x.square())
        lam_y = 2.0 * self.corr_length / (1.0 + scale_y.square())

        lam_2d = lam_x[:, None] * lam_y[None, :]
        flat = lam_2d.reshape(-1)
        top_vals, top_idx = torch.topk(flat, k=self.R, largest=True, sorted=True)

        idx_x = torch.div(top_idx, one_d_modes, rounding_mode="floor")
        idx_y = top_idx % one_d_modes
        mode_x = mode_ids.index_select(0, idx_x)
        mode_y = mode_ids.index_select(0, idx_y)

        return top_vals.to(dtype=torch.float32), mode_x, mode_y

    def _eval_1d_modes(self, coord, mode_idx, bounds):
        coord_vec = self._as_vector(coord, "coord")
        low, high = bounds
        length = high - low
        idx_float = mode_idx.to(device=coord_vec.device, dtype=coord_vec.dtype)
        phase = (
            idx_float.unsqueeze(0) * torch.pi * (coord_vec.unsqueeze(1) - low) / length
        )
        odd = (mode_idx.to(device=coord_vec.device) % 2) == 1
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        norm = math.sqrt(2.0 / length)
        return norm * torch.where(odd.unsqueeze(0), sin_part, cos_part)

    def _eval_1d_mode_derivatives(self, coord, mode_idx, bounds):
        coord_vec = self._as_vector(coord, "coord")
        low, high = bounds
        length = high - low
        idx = mode_idx.to(device=coord_vec.device)
        idx_float = idx.to(dtype=coord_vec.dtype)
        phase = (
            idx_float.unsqueeze(0) * torch.pi * (coord_vec.unsqueeze(1) - low) / length
        )
        freq = idx_float * math.pi / length
        odd = (idx % 2) == 1
        d_odd = freq.unsqueeze(0) * torch.cos(phase)
        d_even = -freq.unsqueeze(0) * torch.sin(phase)
        norm = math.sqrt(2.0 / length)
        return norm * torch.where(odd.unsqueeze(0), d_odd, d_even)

    def _psi_and_derivatives(self, x, y, Z):
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        eigenvalues = self.kl_eigenvalues.to(device=x_vec.device, dtype=x_vec.dtype)
        sqrt_lam = eigenvalues.sqrt().unsqueeze(0)

        phi_x = self._eval_1d_modes(x_vec, self.kl_mode_x, self.x_range)
        phi_y = self._eval_1d_modes(y_vec, self.kl_mode_y, self.y_range)
        dphi_x_dx = self._eval_1d_mode_derivatives(x_vec, self.kl_mode_x, self.x_range)
        dphi_y_dy = self._eval_1d_mode_derivatives(y_vec, self.kl_mode_y, self.y_range)

        phi = phi_x * phi_y
        dphi_dx = dphi_x_dx * phi_y
        dphi_dy = phi_x * dphi_y_dy

        wZ = sqrt_lam * Z
        psi = (wZ * phi).sum(dim=1)
        psi_x = (wZ * dphi_dx).sum(dim=1)
        psi_y = (wZ * dphi_dy).sum(dim=1)

        return psi, psi_x, psi_y

    def diffusion_coefficient(self, x, y, Z):
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        psi, _, _ = self._psi_and_derivatives(x_vec, y_vec, Z)
        log_k = self.mu_k + self.sigma * psi
        return torch.exp(log_k)

    def compute_residual(self, t, x, y, Z, derivs):
        # Steady state PDE: -div(k grad u) = f
        # -k_x u_x - k_y u_y - k (u_xx + u_yy) - f = 0
        du_dx = derivs["du_dx"]
        du_dy = derivs["du_dy"]
        laplacian_u = derivs["laplacian_u"]

        # We don't care about u_t since it's a steady problem
        # But S^2-PINN architecture includes time. We can just ignore u_t.

        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")

        psi, psi_x, psi_y = self._psi_and_derivatives(x_vec, y_vec, Z)
        k = torch.exp(self.mu_k + self.sigma * psi)

        # k_x = k * sigma * psi_x
        k_x = k * self.sigma * psi_x
        k_y = k * self.sigma * psi_y

        f = torch.full_like(k, self.f_value)

        res = -k_x * du_dx - k_y * du_dy - k * laplacian_u - f
        return res

    def sample_z(self, n: int, device, dtype=torch.float32):
        if self.z_distribution == "gaussian":
            return torch.randn(n, self.R, device=device, dtype=dtype)
        else:
            return 2.0 * torch.rand(n, self.R, device=device, dtype=dtype) - 1.0

    def sample_collocation(self, n: int, device, dtype=torch.float32):
        t = self._batch_uniform(n, self.t_range, device=device, dtype=dtype)
        x = self._batch_uniform(n, self.x_range, device=device, dtype=dtype)
        y = self._batch_uniform(n, self.y_range, device=device, dtype=dtype)
        Z = self.sample_z(n, device=device, dtype=dtype)
        return {"t": t, "x": x, "y": y, "Z": Z}

    def sample_boundary(self, n: int, device, dtype=torch.float32):
        t = self._batch_uniform(n, self.t_range, device=device, dtype=dtype)
        x = self._batch_uniform(n, self.x_range, device=device, dtype=dtype).squeeze(1)
        y = self._batch_uniform(n, self.y_range, device=device, dtype=dtype).squeeze(1)

        edge = torch.randint(0, 4, (n,), device=device)
        x_low = torch.full((n,), self.x_range[0], device=device, dtype=dtype)
        x_high = torch.full((n,), self.x_range[1], device=device, dtype=dtype)
        y_low = torch.full((n,), self.y_range[0], device=device, dtype=dtype)
        y_high = torch.full((n,), self.y_range[1], device=device, dtype=dtype)

        x = torch.where(edge == 0, x_low, x)
        x = torch.where(edge == 1, x_high, x)
        y = torch.where(edge == 2, y_low, y)
        y = torch.where(edge == 3, y_high, y)

        Z = self.sample_z(n, device=device, dtype=dtype)
        return {"t": t, "x": x.unsqueeze(1), "y": y.unsqueeze(1), "Z": Z}

    def sample_initial(self, n: int, device, dtype=torch.float32):
        # We enforce u=0 at t=0 to be consistent with steady state Dirichlet
        t = torch.full((n, 1), self.t_range[0], device=device, dtype=dtype)
        x = self._batch_uniform(n, self.x_range, device=device, dtype=dtype)
        y = self._batch_uniform(n, self.y_range, device=device, dtype=dtype)
        Z = self.sample_z(n, device=device, dtype=dtype)
        return {"t": t, "x": x, "y": y, "Z": Z}

    def exact_solution(self, t, x, y, Z):
        # Returning zero as a dummy since exact solution is unknown.
        # This will be handled in a custom eval loop.
        x_vec = self._as_vector(x, "x")
        return torch.zeros_like(x_vec)

    def eval_grid(self, grid_res, t_val, z_val, device, dtype=torch.float32):
        # We just return the grid points. The true evaluation requires the reference tensors.
        x_lin = torch.linspace(
            self.x_range[0], self.x_range[1], grid_res, device=device, dtype=dtype
        )
        y_lin = torch.linspace(
            self.y_range[0], self.y_range[1], grid_res, device=device, dtype=dtype
        )
        x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")

        x = x_grid.reshape(-1, 1)
        y = y_grid.reshape(-1, 1)
        num_points = x.shape[0]
        t = torch.full((num_points, 1), float(t_val), device=device, dtype=dtype)
        Z = torch.zeros((num_points, self.R), device=device, dtype=dtype)

        u_dummy = torch.zeros_like(x)
        return {
            "t": t,
            "x": x,
            "y": y,
            "Z": Z,
            "u_exact": u_dummy,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }


__all__ = ["StochasticDarcyMC2D"]
