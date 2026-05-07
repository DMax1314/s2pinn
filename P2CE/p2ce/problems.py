"""Problem definitions for fair PC^2 comparisons.

The formulas mirror the current S2-PINN PDE classes, but are written in NumPy
because UQpy's physics-informed PCE stack is NumPy based.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Domain:
    x: Tuple[float, float] = (-1.0, 1.0)
    y: Tuple[float, float] = (-1.0, 1.0)
    t: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    problem_type: str
    R: int
    sigma: float
    corr_length: float
    domain: Domain = Domain()
    mu_k: float = 0.0
    gamma: float = 0.3
    f_value: float = 1.0


class BaseProblem:
    """Shared sampling and KL helpers for PC^2 benchmarks."""

    transient: bool = False
    has_exact_solution: bool = False

    def __init__(self, cfg: ProblemConfig):
        if cfg.R <= 0:
            raise ValueError(f"R must be positive, got {cfg.R}")
        if cfg.corr_length <= 0.0:
            raise ValueError(f"corr_length must be positive, got {cfg.corr_length}")
        self.cfg = cfg
        self.R = int(cfg.R)
        self.domain = cfg.domain
        self.kl_eigenvalues, self.kl_mode_x, self.kl_mode_y = self._compute_kl_modes()

    @property
    def deterministic_dim(self) -> int:
        return 3 if self.transient else 2

    @property
    def input_dim(self) -> int:
        return self.deterministic_dim + self.R

    @property
    def lower_bounds(self) -> list[float]:
        bounds = [self.domain.x[0], self.domain.y[0]]
        if self.transient:
            bounds.append(self.domain.t[0])
        return bounds

    @property
    def upper_bounds(self) -> list[float]:
        bounds = [self.domain.x[1], self.domain.y[1]]
        if self.transient:
            bounds.append(self.domain.t[1])
        return bounds

    @property
    def z_slice(self) -> slice:
        return slice(self.deterministic_dim, self.deterministic_dim + self.R)

    def _compute_kl_modes(self) -> Tuple[Array, Array, Array]:
        one_d_modes = int(math.ceil(math.sqrt(float(self.R)))) + 4
        mode_ids = np.arange(1, one_d_modes + 1, dtype=int)
        mode_ids_float = mode_ids.astype(float)

        x_length = self.domain.x[1] - self.domain.x[0]
        y_length = self.domain.y[1] - self.domain.y[0]
        scale_x = mode_ids_float * np.pi * self.cfg.corr_length / x_length
        scale_y = mode_ids_float * np.pi * self.cfg.corr_length / y_length

        lam_x = 2.0 * self.cfg.corr_length / (1.0 + scale_x**2)
        lam_y = 2.0 * self.cfg.corr_length / (1.0 + scale_y**2)
        lam_2d = lam_x[:, None] * lam_y[None, :]
        flat = lam_2d.reshape(-1)
        top_idx = np.argsort(flat)[::-1][: self.R]

        idx_x = top_idx // one_d_modes
        idx_y = top_idx % one_d_modes
        return flat[top_idx], mode_ids[idx_x], mode_ids[idx_y]

    @staticmethod
    def _eval_1d_modes(coord: Array, mode_idx: Array, bounds: Tuple[float, float]) -> Array:
        low, high = bounds
        length = high - low
        phase = mode_idx[None, :] * np.pi * (coord[:, None] - low) / length
        odd = (mode_idx % 2) == 1
        values = np.where(odd[None, :], np.sin(phase), np.cos(phase))
        return math.sqrt(2.0 / length) * values

    @staticmethod
    def _eval_1d_mode_derivatives(
        coord: Array, mode_idx: Array, bounds: Tuple[float, float]
    ) -> Array:
        low, high = bounds
        length = high - low
        phase = mode_idx[None, :] * np.pi * (coord[:, None] - low) / length
        freq = mode_idx.astype(float) * np.pi / length
        odd = (mode_idx % 2) == 1
        values = np.where(
            odd[None, :],
            freq[None, :] * np.cos(phase),
            -freq[None, :] * np.sin(phase),
        )
        return math.sqrt(2.0 / length) * values

    def _phi(self, x: Array, y: Array) -> Array:
        return self._eval_1d_modes(x, self.kl_mode_x, self.domain.x) * self._eval_1d_modes(
            y, self.kl_mode_y, self.domain.y
        )

    def _phi_derivatives(self, x: Array, y: Array) -> Tuple[Array, Array, Array]:
        phi_x = self._eval_1d_modes(x, self.kl_mode_x, self.domain.x)
        phi_y = self._eval_1d_modes(y, self.kl_mode_y, self.domain.y)
        dphi_x_dx = self._eval_1d_mode_derivatives(x, self.kl_mode_x, self.domain.x)
        dphi_y_dy = self._eval_1d_mode_derivatives(y, self.kl_mode_y, self.domain.y)
        return phi_x * phi_y, dphi_x_dx * phi_y, phi_x * dphi_y_dy

    def split(self, sample: Array) -> Tuple[Array, Array, Array | None, Array]:
        sample = np.asarray(sample, dtype=float)
        x = sample[:, 0]
        y = sample[:, 1]
        if self.transient:
            return x, y, sample[:, 2], sample[:, 3 : 3 + self.R]
        return x, y, None, sample[:, 2 : 2 + self.R]

    def standardized_to_physical(self, standardized: Array) -> Array:
        standardized = np.asarray(standardized, dtype=float)
        physical = standardized.copy()
        for j, (low, high) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            physical[:, j] = low + 0.5 * (standardized[:, j] + 1.0) * (high - low)
        return physical

    def derivative_multiplier(self, variable: int, order: int) -> float:
        low = self.lower_bounds[variable]
        high = self.upper_bounds[variable]
        return (2.0 / (high - low)) ** order

    def sample_z(self, rng: np.random.Generator, n: int) -> Array:
        return rng.standard_normal((n, self.R))

    def sample_interior(self, rng: np.random.Generator, n: int) -> Array:
        x = rng.uniform(self.domain.x[0], self.domain.x[1], size=(n, 1))
        y = rng.uniform(self.domain.y[0], self.domain.y[1], size=(n, 1))
        z = self.sample_z(rng, n)
        if self.transient:
            t = rng.uniform(self.domain.t[0], self.domain.t[1], size=(n, 1))
            return np.hstack([x, y, t, z])
        return np.hstack([x, y, z])

    def sample_spatial_boundary(self, rng: np.random.Generator, n: int) -> Array:
        x = rng.uniform(self.domain.x[0], self.domain.x[1], size=n)
        y = rng.uniform(self.domain.y[0], self.domain.y[1], size=n)
        edge = rng.integers(0, 4, size=n)
        x[edge == 0] = self.domain.x[0]
        x[edge == 1] = self.domain.x[1]
        y[edge == 2] = self.domain.y[0]
        y[edge == 3] = self.domain.y[1]
        z = self.sample_z(rng, n)
        if self.transient:
            t = rng.uniform(self.domain.t[0], self.domain.t[1], size=(n, 1))
            return np.hstack([x[:, None], y[:, None], t, z])
        return np.hstack([x[:, None], y[:, None], z])

    def sample_dirichlet(self, rng: np.random.Generator, n_boundary: int, n_initial: int = 0):
        boundary = self.sample_spatial_boundary(rng, n_boundary)
        boundary_values = self.dirichlet_values(boundary)
        if not self.transient or n_initial <= 0:
            return boundary, boundary_values

        x = rng.uniform(self.domain.x[0], self.domain.x[1], size=(n_initial, 1))
        y = rng.uniform(self.domain.y[0], self.domain.y[1], size=(n_initial, 1))
        t = np.full((n_initial, 1), self.domain.t[0])
        z = self.sample_z(rng, n_initial)
        initial = np.hstack([x, y, t, z])
        coords = np.vstack([boundary, initial])
        values = np.concatenate([boundary_values, self.dirichlet_values(initial)])
        return coords, values

    def dirichlet_values(self, sample: Array) -> Array:
        if self.has_exact_solution:
            return self.exact_solution(sample)
        return np.zeros(sample.shape[0])

    def diffusion_coefficient(self, sample: Array) -> Array:
        x, y, _, z = self.split(sample)
        phi = self._phi(x, y)
        weighted = phi * np.sqrt(self.kl_eigenvalues)[None, :]
        log_k = self.cfg.mu_k + self.cfg.sigma * np.sum(weighted * z, axis=1)
        return np.exp(log_k)

    def source_term(self, sample: Array) -> Array:
        raise NotImplementedError

    def exact_solution(self, sample: Array) -> Array:
        raise NotImplementedError


class DiffusionProblem(BaseProblem):
    transient = True
    has_exact_solution = True

    def exact_solution(self, sample: Array) -> Array:
        x, y, t, z = self.split(sample)
        z_amp = 1.0 + 0.1 * np.sum(z, axis=1)
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t) * z_amp

    def source_term(self, sample: Array) -> Array:
        u = self.exact_solution(sample)
        k = self.diffusion_coefficient(sample)
        u_t = -u
        laplacian = -2.0 * (np.pi**2) * u
        return u_t - k * laplacian

    def exact_moments(self, coordinates: Array) -> Tuple[Array, Array]:
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        t = coordinates[:, 2]
        base = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)
        mean = base
        var = (base**2) * 0.01 * self.R
        return mean, var


class PoissonMCProblem(BaseProblem):
    transient = False
    has_exact_solution = False

    def source_term(self, sample: Array) -> Array:
        x, y, _, z = self.split(sample)
        phi = self._phi(x, y)
        weighted = phi * np.sqrt(self.kl_eigenvalues)[None, :]
        stochastic_part = np.sum(weighted * z, axis=1)
        return self.cfg.f_value + self.cfg.sigma * stochastic_part


class DarcyMCProblem(BaseProblem):
    transient = False
    has_exact_solution = False

    def coefficient_and_derivatives(self, sample: Array) -> Tuple[Array, Array, Array]:
        x, y, _, z = self.split(sample)
        phi, dphi_dx, dphi_dy = self._phi_derivatives(x, y)
        weighted = np.sqrt(self.kl_eigenvalues)[None, :] * z
        psi = np.sum(weighted * phi, axis=1)
        psi_x = np.sum(weighted * dphi_dx, axis=1)
        psi_y = np.sum(weighted * dphi_dy, axis=1)
        k = np.exp(self.cfg.mu_k + self.cfg.sigma * psi)
        k_x = k * self.cfg.sigma * psi_x
        k_y = k * self.cfg.sigma * psi_y
        return k, k_x, k_y

    def source_term(self, sample: Array) -> Array:
        return np.full(sample.shape[0], self.cfg.f_value)


def problem_from_dict(raw: Dict) -> BaseProblem:
    domain_raw = raw.get("domain", {})
    domain = Domain(
        x=tuple(domain_raw.get("x", [-1.0, 1.0])),
        y=tuple(domain_raw.get("y", [-1.0, 1.0])),
        t=tuple(domain_raw.get("t", [0.0, 1.0])),
    )
    cfg = ProblemConfig(
        name=raw["name"],
        problem_type=raw["type"],
        R=int(raw.get("R", 5)),
        sigma=float(raw.get("sigma", 1.0)),
        corr_length=float(raw.get("corr_length", 1.0)),
        domain=domain,
        mu_k=float(raw.get("mu_k", 0.0)),
        gamma=float(raw.get("gamma", 0.3)),
        f_value=float(raw.get("f_value", 1.0)),
    )
    if cfg.problem_type == "diffusion":
        return DiffusionProblem(cfg)
    if cfg.problem_type == "poisson_mc":
        return PoissonMCProblem(cfg)
    if cfg.problem_type == "darcy_mc":
        return DarcyMCProblem(cfg)
    raise ValueError(f"Unsupported PC^2 problem type: {cfg.problem_type}")


def grid_coordinates(problem: BaseProblem, grid_res: int, t_value: float = 1.0) -> Array:
    x = np.linspace(problem.domain.x[0], problem.domain.x[1], grid_res)
    y = np.linspace(problem.domain.y[0], problem.domain.y[1], grid_res)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if problem.transient:
        tt = np.full_like(xx, t_value, dtype=float)
        return np.column_stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)])
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def make_full_samples(problem: BaseProblem, deterministic: Array, z: Array) -> Array:
    deterministic = np.asarray(deterministic, dtype=float)
    z = np.asarray(z, dtype=float)
    if deterministic.ndim == 1:
        deterministic = deterministic[None, :]
    if z.ndim == 1:
        z = z[None, :]
    if deterministic.shape[0] == 1 and z.shape[0] > 1:
        deterministic = np.repeat(deterministic, z.shape[0], axis=0)
    if z.shape[0] == 1 and deterministic.shape[0] > 1:
        z = np.repeat(z, deterministic.shape[0], axis=0)
    if deterministic.shape[0] != z.shape[0]:
        raise ValueError("deterministic and z sample counts do not match")
    return np.hstack([deterministic, z])


def iter_chunks(indices: Iterable[int], chunk_size: int) -> Iterable[list[int]]:
    chunk: list[int] = []
    for idx in indices:
        chunk.append(idx)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
