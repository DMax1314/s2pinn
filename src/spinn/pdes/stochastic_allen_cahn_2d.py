import math
from typing import Dict, Tuple, Union

import torch


class StochasticAllenCahn2D:
    """Stochastic Allen-Cahn equation on (x,y) ∈ [-1,1]², t ∈ [0,1].

    PDE: ∂u/∂t = k(x,y,Z) Δu + u − u³ + s(t,x,y,Z)

    where k is a KL-lognormal random diffusivity and the source s is
    manufactured so that the exact solution is known analytically.

    Exact solution:
        u(t,x,y,Z) = sin(πx) sin(πy) exp(−t) (1 + 0.1 Σ Zᵢ)

    The nonlinear cubic reaction term (u − u³) distinguishes this from
    the linear stochastic diffusion benchmark by testing the model's
    ability to capture nonlinear PDE dynamics under parametric uncertainty.
    """

    def __init__(
        self,
        R: int = 3,
        sigma: float = 1.0,
        corr_length: float = 1.0,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        mu_k: float = 0.0,
        z_distribution: str = "gaussian",
    ):
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if corr_length <= 0.0:
            raise ValueError(f"corr_length must be positive, got {corr_length}")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"x_range must satisfy min < max, got {x_range}")
        if y_range[0] >= y_range[1]:
            raise ValueError(f"y_range must satisfy min < max, got {y_range}")
        if t_range[0] >= t_range[1]:
            raise ValueError(f"t_range must satisfy min < max, got {t_range}")
        if z_distribution not in {"gaussian", "uniform"}:
            raise ValueError(
                f"z_distribution must be 'gaussian' or 'uniform', got {z_distribution}"
            )

        self.R = int(R)
        self.sigma = float(sigma)
        self.corr_length = float(corr_length)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.t_range = (float(t_range[0]), float(t_range[1]))
        self.mu_k = float(mu_k)
        self.z_distribution = z_distribution

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
        raise ValueError(
            f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}"
        )

    @staticmethod
    def _batch_uniform(
        n: int,
        bounds: Tuple[float, float],
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1, device=device, dtype=dtype)

    def _compute_kl_modes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _eval_1d_modes(
        self,
        coord: torch.Tensor,
        mode_idx: torch.Tensor,
        bounds: Tuple[float, float],
    ) -> torch.Tensor:
        coord_vec = self._as_vector(coord, "coord")
        low, high = bounds
        length = high - low
        if length <= 0.0:
            raise ValueError(f"bounds must satisfy min < max, got {bounds}")

        idx = mode_idx.to(device=coord_vec.device)
        idx_float = idx.to(dtype=coord_vec.dtype)
        phase = (
            idx_float.unsqueeze(0) * torch.pi * (coord_vec.unsqueeze(1) - low) / length
        )

        odd = (idx % 2) == 1
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        norm = math.sqrt(2.0 / length)
        return norm * torch.where(odd.unsqueeze(0), sin_part, cos_part)

    def _prepare_xy_z(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        if x_vec.shape != y_vec.shape:
            raise ValueError(
                f"x and y must match shape, got {tuple(x_vec.shape)} and {tuple(y_vec.shape)}"
            )
        if Z.ndim != 2 or Z.shape[1] != self.R:
            raise ValueError(f"Z must have shape [B, {self.R}], got {tuple(Z.shape)}")
        if Z.shape[0] != x_vec.shape[0]:
            raise ValueError(
                f"Batch mismatch: x/y have {x_vec.shape[0]} samples, Z has {Z.shape[0]}"
            )
        return x_vec, y_vec, Z

    def _prepare_txy_z(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_vec = self._as_vector(t, "t")
        x_vec, y_vec, Z_checked = self._prepare_xy_z(x, y, Z)
        if t_vec.shape[0] != x_vec.shape[0]:
            raise ValueError(
                f"Batch mismatch: t has {t_vec.shape[0]} samples, x/y have {x_vec.shape[0]}"
            )
        return t_vec, x_vec, y_vec, Z_checked

    def kl_eigenfunctions(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        if x_vec.shape != y_vec.shape:
            raise ValueError(
                f"x and y must match shape, got {tuple(x_vec.shape)} and {tuple(y_vec.shape)}"
            )

        mode_x = self.kl_mode_x.to(device=x_vec.device)
        mode_y = self.kl_mode_y.to(device=y_vec.device)
        phi_x = self._eval_1d_modes(x_vec, mode_x, self.x_range)
        phi_y = self._eval_1d_modes(y_vec, mode_y, self.y_range)
        phi = phi_x * phi_y

        assert phi.shape == (x_vec.shape[0], self.R)
        return phi

    def diffusion_coefficient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        x_vec, y_vec, Z_checked = self._prepare_xy_z(x, y, Z)
        phi = self.kl_eigenfunctions(x_vec, y_vec)
        eigenvalues = self.kl_eigenvalues.to(device=phi.device, dtype=phi.dtype)

        weighted_modes = phi * eigenvalues.sqrt().unsqueeze(0)
        log_k = self.mu_k + self.sigma * (weighted_modes * Z_checked).sum(dim=1)
        k = torch.exp(log_k)

        assert k.shape == (x_vec.shape[0],)
        return k

    def exact_solution(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        t_vec, x_vec, y_vec, Z_checked = self._prepare_txy_z(t, x, y, Z)
        z_amp = 1.0 + 0.1 * Z_checked.sum(dim=1)

        u = (
            torch.sin(torch.pi * x_vec)
            * torch.sin(torch.pi * y_vec)
            * torch.exp(-t_vec)
            * z_amp
        )

        assert u.shape == (t_vec.shape[0],)
        return u

    def source_term(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """Manufactured source: s = u_t − k Δu − (u − u³)."""
        t_vec, x_vec, y_vec, Z_checked = self._prepare_txy_z(t, x, y, Z)
        u_exact = self.exact_solution(t_vec, x_vec, y_vec, Z_checked)
        k = self.diffusion_coefficient(x_vec, y_vec, Z_checked)

        u_t = -u_exact
        laplacian_u = -2.0 * (torch.pi**2) * u_exact
        reaction = u_exact - u_exact**3

        s = u_t - k * laplacian_u - reaction

        assert s.shape == (t_vec.shape[0],)
        return s

    def compute_residual(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
        derivs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Residual: u_t − k Δu − (u − u³) − s."""
        u = derivs["u"]
        u_t = derivs["u_t"]
        laplacian_u = derivs["laplacian_u"]
        k = self.diffusion_coefficient(x, y, Z)
        s = self.source_term(t, x, y, Z)
        reaction = u - u**3
        return u_t - k * laplacian_u - reaction - s

    def sample_z(
        self,
        n: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        if self.z_distribution == "gaussian":
            Z = torch.randn(n, self.R, device=device, dtype=dtype)
        else:
            Z = 2.0 * torch.rand(n, self.R, device=device, dtype=dtype) - 1.0

        assert Z.shape == (n, self.R)
        return Z

    def sample_collocation(
        self,
        n: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        t = self._batch_uniform(n, self.t_range, device=device, dtype=dtype)
        x = self._batch_uniform(n, self.x_range, device=device, dtype=dtype)
        y = self._batch_uniform(n, self.y_range, device=device, dtype=dtype)
        Z = self.sample_z(n, device=device, dtype=dtype)
        return {"t": t, "x": x, "y": y, "Z": Z}

    def sample_boundary(
        self,
        n: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

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

    def sample_initial(
        self,
        n: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        t = torch.full((n, 1), self.t_range[0], device=device, dtype=dtype)
        x = self._batch_uniform(n, self.x_range, device=device, dtype=dtype)
        y = self._batch_uniform(n, self.y_range, device=device, dtype=dtype)
        Z = self.sample_z(n, device=device, dtype=dtype)
        return {"t": t, "x": x, "y": y, "Z": Z}

    def eval_grid(
        self,
        grid_res: int,
        t_val: float,
        z_val: torch.Tensor,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        if grid_res <= 1:
            raise ValueError(f"grid_res must be >= 2, got {grid_res}")

        x_lin = torch.linspace(
            self.x_range[0],
            self.x_range[1],
            grid_res,
            device=device,
            dtype=dtype,
        )
        y_lin = torch.linspace(
            self.y_range[0],
            self.y_range[1],
            grid_res,
            device=device,
            dtype=dtype,
        )
        x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")

        x = x_grid.reshape(-1, 1)
        y = y_grid.reshape(-1, 1)
        num_points = x.shape[0]
        t = torch.full((num_points, 1), float(t_val), device=device, dtype=dtype)

        z_tensor = z_val.to(device=device, dtype=dtype)
        if z_tensor.ndim == 1:
            if z_tensor.shape[0] != self.R:
                raise ValueError(
                    f"z_val must have length {self.R}, got {z_tensor.shape[0]}"
                )
            Z = z_tensor.unsqueeze(0).expand(num_points, -1)
        elif z_tensor.ndim == 2:
            if z_tensor.shape[1] != self.R:
                raise ValueError(
                    f"z_val must have shape [1, {self.R}] or [N, {self.R}], got {tuple(z_tensor.shape)}"
                )
            if z_tensor.shape[0] == 1:
                Z = z_tensor.expand(num_points, -1)
            elif z_tensor.shape[0] == num_points:
                Z = z_tensor
            else:
                raise ValueError(
                    f"z_val first dim must be 1 or {num_points}, got {z_tensor.shape[0]}"
                )
        else:
            raise ValueError(f"z_val must have shape [{self.R}] or [N, {self.R}]")

        u_exact = self.exact_solution(t, x, y, Z).unsqueeze(1)
        return {
            "t": t,
            "x": x,
            "y": y,
            "Z": Z,
            "u_exact": u_exact,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }


__all__ = ["StochasticAllenCahn2D"]
