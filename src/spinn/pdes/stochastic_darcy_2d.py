"""Stochastic Darcy flow (divergence-form diffusion) on 2D spatial domain.

PDE:  u_t − div(k(x,y,ω) ∇u) = f    on  Ω × (0,T]
      u  = u_exact                   on ∂Ω × (0,T]
      u(0,·,·,ω) = u_exact(0,·,·)   on  Ω

Diffusion coefficient (log-normal via KL expansion):
    k(x,y,Z) = exp(μ_k + σ Σᵢ √λᵢ Zᵢ φᵢ(x,y))

Manufactured solution with *nonlinear* stochastic dependence:
    u(t,x,y,Z) = sin(πx) sin(πy) exp(−t) · exp(γ Σᵢ √λᵢ Zᵢ φᵢ(x,y))

Denoting  ψ(x,y,Z) = γ Σᵢ √λᵢ Zᵢ φᵢ(x,y)  and  H = exp(ψ),
the solution is  u = S · T · H  where S = sin(πx)sin(πy), T = exp(−t).

Key design choices
------------------
1.  **Divergence form**: −div(k∇u) produces cross-terms dk/dx · du/dx that
    couple the random coefficient's spatial variation with the solution
    gradient.  This is the standard UQ benchmark operator (Darcy / random
    coefficient diffusion).
2.  **Nonlinear Z-dependence**: The factor H = exp(ψ) = k^{γ/σ} makes the
    solution depend on Z through an exponential of the KL field.  The
    parameter γ controls difficulty:
        γ → 0 : solution becomes Z-independent (trivial)
        γ ≈ σ : solution varies as strongly as k itself
    Default γ = 0.3 gives moderate but genuinely nonlinear Z-dependence.
3.  **Analytically computable source**:  Because u is a manufactured solution,
    f = u_t − div(k∇u) is known in closed form (though complex), enabling
    exact evaluation and rel-L2 metrics.

Mathematical details
--------------------
Let  α = (σ + γ)/γ   (so  χ_x = d(log k)/dx = (σ/γ) ψ_x,  χ_x + ψ_x = α ψ_x).

    div(k∇u) = k · T · H · [
        (α+1) (S_x ψ_x + S_y ψ_y)
      + α S (ψ_x² + ψ_y²)
      + (S_xx + S_yy)
      + S (ψ_xx + ψ_yy)
    ]

    f = u_t − div(k∇u)  = −u − div(k∇u)

Residual (model outputs):
    res = u_t − k_x du/dx − k_y du/dy − k Δu − f
"""

import math
from typing import Dict, Tuple, Union

import torch


class StochasticDarcy2D:
    """Stochastic Darcy flow on (x,y) ∈ [-1,1]², t ∈ [0,1].

    Args:
        R: Number of random dimensions (KL terms).
        sigma: Standard deviation of the log-k field.
        corr_length: Correlation length of the KL expansion.
        gamma: Nonlinearity strength for the manufactured solution.
            Controls how strongly Z enters the solution (via exp(γ·KL_field)).
            Must be > 0.
        x_range: Spatial domain bounds in x.
        y_range: Spatial domain bounds in y.
        t_range: Temporal domain bounds.
        mu_k: Mean of the log-k field (constant shift).
        z_distribution: Distribution for Z samples ('gaussian' or 'uniform').
    """

    def __init__(
        self,
        R: int = 3,
        sigma: float = 1.0,
        corr_length: float = 1.0,
        gamma: float = 0.3,
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
        if gamma <= 0.0:
            raise ValueError(
                f"gamma must be positive for Darcy PDE (controls nonlinear "
                f"Z-dependence), got {gamma}.  Use StochasticDiffusion2D for "
                f"the linear-Z case."
            )
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
        self.gamma = float(gamma)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.t_range = (float(t_range[0]), float(t_range[1]))
        self.mu_k = float(mu_k)
        self.z_distribution = z_distribution

        kl_eigenvalues, kl_mode_x, kl_mode_y = self._compute_kl_modes()
        self.kl_eigenvalues = kl_eigenvalues
        self.kl_mode_x = kl_mode_x
        self.kl_mode_y = kl_mode_y

    # ------------------------------------------------------------------
    #  Utility helpers (identical to StochasticDiffusion2D)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    #  KL expansion
    # ------------------------------------------------------------------

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
        """Evaluate 1-D KL eigenfunctions φ_m(x).  Shape: [B, R]."""
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

    def _eval_1d_mode_derivatives(
        self,
        coord: torch.Tensor,
        mode_idx: torch.Tensor,
        bounds: Tuple[float, float],
    ) -> torch.Tensor:
        """First spatial derivative of 1-D KL modes dφ_m/dx.  Shape: [B, R].

        Odd mode  (sin): dφ/dx =  (m π / L) · norm · cos(phase)
        Even mode (cos): dφ/dx = −(m π / L) · norm · sin(phase)
        """
        coord_vec = self._as_vector(coord, "coord")
        low, high = bounds
        length = high - low

        idx = mode_idx.to(device=coord_vec.device)
        idx_float = idx.to(dtype=coord_vec.dtype)
        phase = (
            idx_float.unsqueeze(0) * torch.pi * (coord_vec.unsqueeze(1) - low) / length
        )
        freq = idx_float * math.pi / length  # [R]

        odd = (idx % 2) == 1
        # sin → d/dx = +freq · cos;   cos → d/dx = −freq · sin
        d_odd = freq.unsqueeze(0) * torch.cos(phase)
        d_even = -freq.unsqueeze(0) * torch.sin(phase)
        norm = math.sqrt(2.0 / length)
        return norm * torch.where(odd.unsqueeze(0), d_odd, d_even)

    def _eval_1d_mode_second_derivatives(
        self,
        coord: torch.Tensor,
        mode_idx: torch.Tensor,
        bounds: Tuple[float, float],
    ) -> torch.Tensor:
        """Second spatial derivative d²φ_m/dx².  Shape: [B, R].

        For both sin and cos modes:  d²φ/dx² = −(m π / L)² φ.
        """
        coord_vec = self._as_vector(coord, "coord")
        phi = self._eval_1d_modes(coord_vec, mode_idx, bounds)  # [B, R]
        idx_float = mode_idx.to(device=coord_vec.device, dtype=coord_vec.dtype)
        low, high = bounds
        length = high - low
        freq_sq = (idx_float * math.pi / length) ** 2  # [R]
        return -freq_sq.unsqueeze(0) * phi  # [B, R]

    # ------------------------------------------------------------------
    #  ψ field and its spatial derivatives
    # ------------------------------------------------------------------

    def _psi_and_derivatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ψ = γ Σ √λᵢ Zᵢ φᵢ(x,y) and spatial derivatives.

        Returns:
            (ψ, ψ_x, ψ_y, ψ_xx, ψ_yy)  each [B].
        """
        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")

        eigenvalues = self.kl_eigenvalues.to(device=x_vec.device, dtype=x_vec.dtype)
        sqrt_lam = eigenvalues.sqrt().unsqueeze(0)  # [1, R]

        # 1-D evaluations
        phi_x = self._eval_1d_modes(x_vec, self.kl_mode_x, self.x_range)  # [B, R]
        phi_y = self._eval_1d_modes(y_vec, self.kl_mode_y, self.y_range)  # [B, R]
        dphi_x_dx = self._eval_1d_mode_derivatives(x_vec, self.kl_mode_x, self.x_range)
        dphi_y_dy = self._eval_1d_mode_derivatives(y_vec, self.kl_mode_y, self.y_range)
        d2phi_x_dx2 = self._eval_1d_mode_second_derivatives(
            x_vec, self.kl_mode_x, self.x_range
        )
        d2phi_y_dy2 = self._eval_1d_mode_second_derivatives(
            y_vec, self.kl_mode_y, self.y_range
        )

        # 2-D mode products
        phi = phi_x * phi_y  # φᵢ(x,y)
        dphi_dx = dphi_x_dx * phi_y  # ∂φᵢ/∂x
        dphi_dy = phi_x * dphi_y_dy  # ∂φᵢ/∂y
        d2phi_dx2 = d2phi_x_dx2 * phi_y  # ∂²φᵢ/∂x²
        d2phi_dy2 = phi_x * d2phi_y_dy2  # ∂²φᵢ/∂y²

        # Weighted sum  √λᵢ · Zᵢ · (·)
        wZ = sqrt_lam * Z  # [B, R]

        psi = self.gamma * (wZ * phi).sum(dim=1)
        psi_x = self.gamma * (wZ * dphi_dx).sum(dim=1)
        psi_y = self.gamma * (wZ * dphi_dy).sum(dim=1)
        psi_xx = self.gamma * (wZ * d2phi_dx2).sum(dim=1)
        psi_yy = self.gamma * (wZ * d2phi_dy2).sum(dim=1)

        return psi, psi_x, psi_y, psi_xx, psi_yy

    # ------------------------------------------------------------------
    #  Shared PDE interface
    # ------------------------------------------------------------------

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
        """k(x,y,Z) = exp(μ_k + σ Σ √λᵢ Zᵢ φᵢ)."""
        x_vec, y_vec, Z_checked = self._prepare_xy_z(x, y, Z)
        phi = self.kl_eigenfunctions(x_vec, y_vec)
        eigenvalues = self.kl_eigenvalues.to(device=phi.device, dtype=phi.dtype)

        weighted_modes = phi * eigenvalues.sqrt().unsqueeze(0)
        log_k = self.mu_k + self.sigma * (weighted_modes * Z_checked).sum(dim=1)
        k = torch.exp(log_k)

        assert k.shape == (x_vec.shape[0],)
        return k

    # ------------------------------------------------------------------
    #  Manufactured solution & source
    # ------------------------------------------------------------------

    def exact_solution(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """u = sin(πx) sin(πy) exp(−t) · exp(ψ(x,y,Z))."""
        t_vec, x_vec, y_vec, Z_checked = self._prepare_txy_z(t, x, y, Z)
        psi, _, _, _, _ = self._psi_and_derivatives(x_vec, y_vec, Z_checked)

        S = torch.sin(torch.pi * x_vec) * torch.sin(torch.pi * y_vec)
        T = torch.exp(-t_vec)
        H = torch.exp(psi)
        u = S * T * H

        assert u.shape == (t_vec.shape[0],)
        return u

    def source_term(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """Manufactured source f = u_t − div(k ∇u)."""
        t_vec, x_vec, y_vec, Z_checked = self._prepare_txy_z(t, x, y, Z)
        psi, psi_x, psi_y, psi_xx, psi_yy = self._psi_and_derivatives(
            x_vec, y_vec, Z_checked
        )

        S = torch.sin(torch.pi * x_vec) * torch.sin(torch.pi * y_vec)
        S_x = torch.pi * torch.cos(torch.pi * x_vec) * torch.sin(torch.pi * y_vec)
        S_y = torch.sin(torch.pi * x_vec) * torch.pi * torch.cos(torch.pi * y_vec)
        S_xx = -(torch.pi**2) * S
        S_yy = -(torch.pi**2) * S
        T = torch.exp(-t_vec)
        H = torch.exp(psi)
        k = self.diffusion_coefficient(x_vec, y_vec, Z_checked)

        u = S * T * H
        u_t = -u

        # α = (σ + γ) / γ
        alpha = (self.sigma + self.gamma) / self.gamma

        # div(k∇u) = k·T·H · [ (α+1)(S_x ψ_x + S_y ψ_y)
        #                      + α S (ψ_x² + ψ_y²)
        #                      + (S_xx + S_yy)
        #                      + S (ψ_xx + ψ_yy) ]
        div_k_grad_u = (
            k
            * T
            * H
            * (
                (alpha + 1.0) * (S_x * psi_x + S_y * psi_y)
                + alpha * S * (psi_x**2 + psi_y**2)
                + (S_xx + S_yy)
                + S * (psi_xx + psi_yy)
            )
        )

        f = u_t - div_k_grad_u

        assert f.shape == (t_vec.shape[0],)
        return f

    # ------------------------------------------------------------------
    #  PDE residual (used by training loop)
    # ------------------------------------------------------------------

    def compute_residual(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor,
        derivs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Residual: u_t − k_x du/dx − k_y du/dy − k Δu − f.

        Computes k, dk/dx, dk/dy, and the source f in a single pass
        to avoid redundant KL eigenfunction evaluations.

        Args:
            t: Time coordinates [B].
            x: Spatial x-coordinates [B].
            y: Spatial y-coordinates [B].
            Z: Stochastic coordinates [B, R].
            derivs: Model derivative dict with keys
                {u, u_t, du_dx, du_dy, laplacian_u}.

        Returns:
            Residual tensor [B].
        """
        u_t_model = derivs["u_t"]
        du_dx = derivs["du_dx"]
        du_dy = derivs["du_dy"]
        laplacian_u = derivs["laplacian_u"]

        x_vec = self._as_vector(x, "x")
        y_vec = self._as_vector(y, "y")
        t_vec = self._as_vector(t, "t")

        # ---- shared quantities ----
        k = self.diffusion_coefficient(x_vec, y_vec, Z)
        psi, psi_x, psi_y, psi_xx, psi_yy = self._psi_and_derivatives(x_vec, y_vec, Z)

        # dk/dx = k · χ_x = k · (σ/γ) ψ_x   (and similarly for y)
        ratio = self.sigma / self.gamma
        k_x = k * ratio * psi_x
        k_y = k * ratio * psi_y

        # ---- source term (inline) ----
        S = torch.sin(torch.pi * x_vec) * torch.sin(torch.pi * y_vec)
        S_x = torch.pi * torch.cos(torch.pi * x_vec) * torch.sin(torch.pi * y_vec)
        S_y = torch.sin(torch.pi * x_vec) * torch.pi * torch.cos(torch.pi * y_vec)
        S_xx = -(torch.pi**2) * S
        S_yy = -(torch.pi**2) * S
        T = torch.exp(-t_vec)
        H = torch.exp(psi)

        u_exact = S * T * H
        u_t_exact = -u_exact

        alpha = (self.sigma + self.gamma) / self.gamma

        div_k_grad_u_exact = (
            k
            * T
            * H
            * (
                (alpha + 1.0) * (S_x * psi_x + S_y * psi_y)
                + alpha * S * (psi_x**2 + psi_y**2)
                + (S_xx + S_yy)
                + S * (psi_xx + psi_yy)
            )
        )

        f = u_t_exact - div_k_grad_u_exact

        # ---- residual for model ----
        return u_t_model - k_x * du_dx - k_y * du_dy - k * laplacian_u - f

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

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


__all__ = ["StochasticDarcy2D"]
