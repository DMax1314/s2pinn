"""FEniCS (dolfinx 0.9) reference solver for stochastic Darcy flow.

PDE:  u_t − div(k(x,y,ω) ∇u) = f   on Ω=(-1,1)², t ∈ (0, T]

Diffusion coefficient:
    k(x,y,Z) = exp(μ_k + σ Σᵢ √λᵢ Zᵢ φᵢ(x,y))

Manufactured solution:
    u(t,x,y,Z) = sin(πx) sin(πy) exp(−t) · exp(γ Σᵢ √λᵢ Zᵢ φᵢ(x,y))

Two modes:
    1. Deterministic (rand_R=0): single Z=0 solve, saves snapshots
    2. Stochastic (rand_R>0): tensor-product quadrature over Z, saves E[u], Var[u]

Usage:
    # Deterministic reference at mesh=64
    python darcy2d_ref.py --mesh 64 --output darcy_det.npz

    # Stochastic reference: R=3, quadrature order p=5
    python darcy2d_ref.py --mesh 64 --rand_R 3 --rand_p 5 --output darcy_uq.npz
"""

import argparse
import math
from typing import List, Tuple

import numpy as np

try:
    from mpi4py import MPI
    import ufl
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    raise ImportError("FEniCS/dolfinx required for reference solver.") from e


# =========================================================================
#  KL expansion (matches StochasticDarcy2D exactly)
# =========================================================================


def compute_kl_modes(
    R: int,
    corr_length: float = 1.0,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eigenvalues[R], mode_x[R], mode_y[R]) matching the PyTorch PDE class."""
    one_d_modes = int(math.ceil(math.sqrt(float(R)))) + 4
    mode_ids = np.arange(1, one_d_modes + 1)

    x_length = x_range[1] - x_range[0]
    y_length = y_range[1] - y_range[0]
    scale_x = mode_ids * np.pi * corr_length / x_length
    scale_y = mode_ids * np.pi * corr_length / y_length

    lam_x = 2.0 * corr_length / (1.0 + scale_x**2)
    lam_y = 2.0 * corr_length / (1.0 + scale_y**2)

    lam_2d = lam_x[:, None] * lam_y[None, :]
    flat = lam_2d.reshape(-1)
    top_idx = np.argsort(-flat)[:R]
    top_vals = flat[top_idx]

    idx_x = top_idx // one_d_modes
    idx_y = top_idx % one_d_modes
    mode_x_ids = mode_ids[idx_x]  # 1-based
    mode_y_ids = mode_ids[idx_y]

    return top_vals, mode_x_ids, mode_y_ids


def eval_1d_mode(
    coord: np.ndarray,
    mode_id: int,
    bounds: Tuple[float, float],
) -> np.ndarray:
    """Evaluate single 1-D KL eigenfunction φ_m(x).

    Odd mode (m odd): sin(m π (x - low) / L) · norm
    Even mode (m even): cos(m π (x - low) / L) · norm
    """
    low, high = bounds
    length = high - low
    phase = mode_id * np.pi * (coord - low) / length
    norm = np.sqrt(2.0 / length)
    if mode_id % 2 == 1:
        return norm * np.sin(phase)
    else:
        return norm * np.cos(phase)


def eval_kl_field(
    x: np.ndarray,
    y: np.ndarray,
    z_vec: np.ndarray,
    eigenvalues: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    sigma: float = 1.0,
    mu_k: float = 0.0,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Evaluate KL field: σ Σ √λᵢ Zᵢ φᵢ(x,y).

    Returns log_k = μ_k + σ Σ √λᵢ Zᵢ φᵢ(x,y).
    """
    R = len(eigenvalues)
    result = np.full_like(x, mu_k, dtype=np.float64)
    sqrt_lam = np.sqrt(eigenvalues)
    for i in range(R):
        phi_x = eval_1d_mode(x, int(mode_x[i]), x_range)
        phi_y = eval_1d_mode(y, int(mode_y[i]), y_range)
        result += sigma * sqrt_lam[i] * z_vec[i] * phi_x * phi_y
    return result


def eval_psi_field(
    x: np.ndarray,
    y: np.ndarray,
    z_vec: np.ndarray,
    eigenvalues: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    gamma: float = 0.3,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Evaluate ψ field: γ Σ √λᵢ Zᵢ φᵢ(x,y)."""
    R = len(eigenvalues)
    result = np.zeros_like(x, dtype=np.float64)
    sqrt_lam = np.sqrt(eigenvalues)
    for i in range(R):
        phi_x = eval_1d_mode(x, int(mode_x[i]), x_range)
        phi_y = eval_1d_mode(y, int(mode_y[i]), y_range)
        result += gamma * sqrt_lam[i] * z_vec[i] * phi_x * phi_y
    return result


# =========================================================================
#  Manufactured solution & source for Darcy
# =========================================================================


def manufactured_solution(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    z_vec: np.ndarray,
    eigenvalues: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    gamma: float = 0.3,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """u = sin(πx) sin(πy) exp(−t) exp(ψ(x,y,Z))."""
    psi = eval_psi_field(
        x, y, z_vec, eigenvalues, mode_x, mode_y, gamma, x_range, y_range
    )
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    return S * np.exp(-t) * np.exp(psi)


def diffusion_coeff(
    x: np.ndarray,
    y: np.ndarray,
    z_vec: np.ndarray,
    eigenvalues: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    sigma: float = 1.0,
    mu_k: float = 0.0,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """k = exp(μ_k + σ Σ √λᵢ Zᵢ φᵢ)."""
    log_k = eval_kl_field(
        x, y, z_vec, eigenvalues, mode_x, mode_y, sigma, mu_k, x_range, y_range
    )
    return np.exp(log_k)


def source_term_darcy(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    z_vec: np.ndarray,
    eigenvalues: np.ndarray,
    mode_x: np.ndarray,
    mode_y: np.ndarray,
    sigma: float = 1.0,
    gamma: float = 0.3,
    mu_k: float = 0.0,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Manufactured source f = u_t − div(k ∇u).

    Uses the same analytical derivation as StochasticDarcy2D.source_term().
    """
    R = len(eigenvalues)
    sqrt_lam = np.sqrt(eigenvalues)

    # ψ and its derivatives
    psi = np.zeros_like(x, dtype=np.float64)
    psi_x = np.zeros_like(x, dtype=np.float64)
    psi_y = np.zeros_like(x, dtype=np.float64)
    psi_xx = np.zeros_like(x, dtype=np.float64)
    psi_yy = np.zeros_like(x, dtype=np.float64)

    for i in range(R):
        mx = int(mode_x[i])
        my = int(mode_y[i])
        lo_x, hi_x = x_range
        lo_y, hi_y = y_range
        Lx = hi_x - lo_x
        Ly = hi_y - lo_y
        norm_x = np.sqrt(2.0 / Lx)
        norm_y = np.sqrt(2.0 / Ly)

        phase_x = mx * np.pi * (x - lo_x) / Lx
        phase_y = my * np.pi * (y - lo_y) / Ly
        freq_x = mx * np.pi / Lx
        freq_y = my * np.pi / Ly

        # φ_x(x), dφ_x/dx, d²φ_x/dx²
        if mx % 2 == 1:
            phi_x_val = norm_x * np.sin(phase_x)
            dphi_x_val = norm_x * freq_x * np.cos(phase_x)
        else:
            phi_x_val = norm_x * np.cos(phase_x)
            dphi_x_val = -norm_x * freq_x * np.sin(phase_x)
        d2phi_x_val = -(freq_x**2) * phi_x_val

        if my % 2 == 1:
            phi_y_val = norm_y * np.sin(phase_y)
            dphi_y_val = norm_y * freq_y * np.cos(phase_y)
        else:
            phi_y_val = norm_y * np.cos(phase_y)
            dphi_y_val = -norm_y * freq_y * np.sin(phase_y)
        d2phi_y_val = -(freq_y**2) * phi_y_val

        wz = gamma * sqrt_lam[i] * z_vec[i]
        phi_2d = phi_x_val * phi_y_val
        psi += wz * phi_2d
        psi_x += wz * dphi_x_val * phi_y_val
        psi_y += wz * phi_x_val * dphi_y_val
        psi_xx += wz * d2phi_x_val * phi_y_val
        psi_yy += wz * phi_x_val * d2phi_y_val

    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    S_x = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    S_y = np.sin(np.pi * x) * np.pi * np.cos(np.pi * y)
    S_xx = -(np.pi**2) * S
    S_yy = -(np.pi**2) * S
    T = np.exp(-t)
    H = np.exp(psi)
    k = diffusion_coeff(
        x, y, z_vec, eigenvalues, mode_x, mode_y, sigma, mu_k, x_range, y_range
    )

    u = S * T * H
    u_t = -u

    alpha = (sigma + gamma) / gamma

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

    return u_t - div_k_grad_u


# =========================================================================
#  Quadrature (reused from diffusion2d_ref.py)
# =========================================================================


def hermite_nodes_weights(n: int):
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(n)
    z = x / np.sqrt(2.0)
    w_prob = w / np.sqrt(np.pi)
    return z, w_prob


def legendre_nodes_weights(n: int):
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(n)
    return x, w * 0.5


def tensorize(nodes_1d: List[np.ndarray], weights_1d: List[np.ndarray]):
    grids = np.meshgrid(*nodes_1d, indexing="ij")
    ws = np.meshgrid(*weights_1d, indexing="ij")
    Z = np.stack([g.reshape(-1) for g in grids], axis=1)
    w = np.prod(np.stack([wi.reshape(-1) for wi in ws], axis=0), axis=0)
    return Z, w


def build_quadrature(R: int, p: int, dist: str = "normal"):
    if R <= 0:
        return np.zeros((0,)), np.zeros((0,))
    n1d = p + 1
    if dist == "normal":
        z, w = hermite_nodes_weights(n1d)
        nodes_1d = [z] * R
        weights_1d = [w] * R
    elif dist == "uniform":
        x, w = legendre_nodes_weights(n1d)
        nodes_1d = [x] * R
        weights_1d = [w] * R
    else:
        raise ValueError("dist must be 'normal' or 'uniform'")
    return tensorize(nodes_1d, weights_1d)


# =========================================================================
#  FEM solver
# =========================================================================


def solve_darcy2d(
    mesh_size: int = 32,
    t_final: float = 1.0,
    num_steps: int = 50,
    rand_R: int = 0,
    rand_p: int = 0,
    quadrature: str = "tensor",
    dist: str = "normal",
    sigma: float = 0.5,
    gamma: float = 0.3,
    mu_k: float = 0.0,
    corr_length: float = 1.0,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    output: str = "darcy2d_ref.npz",
):
    """Solve stochastic Darcy flow with FEniCS.

    Weak form (backward Euler):
        (u^{n+1}/dt, v) + (k ∇u^{n+1}, ∇v) = (u^n/dt + f^{n+1}, v)

    For each Z sample: interpolate k(x,y,Z) and f(t,x,y,Z) onto mesh,
    solve the linear system, accumulate E[u] and E[u²].
    """
    comm = MPI.COMM_WORLD

    # Scale domain: dolfinx create_rectangle uses corners
    domain = mesh.create_rectangle(
        comm,
        [
            list(x_range[0:1]) + list(y_range[0:1]),
            list(x_range[1:2]) + list(y_range[1:2]),
        ],
        [mesh_size, mesh_size],
    )

    V = fem.functionspace(domain, ("Lagrange", 1))

    # Boundary DOFs
    def on_boundary(x_arr):
        return (
            np.isclose(x_arr[0], x_range[0])
            | np.isclose(x_arr[0], x_range[1])
            | np.isclose(x_arr[1], y_range[0])
            | np.isclose(x_arr[1], y_range[1])
        )

    bd_dofs = fem.locate_dofs_geometrical(V, on_boundary)

    dt = float(t_final) / int(num_steps)

    u_prev = fem.Function(V, name="u_prev")
    gD = fem.Function(V, name="gD")
    k_func = fem.Function(V, name="k")
    s_func = fem.Function(V, name="source")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Time-stepping snapshots: t = {0, T/2, T}
    save_steps = sorted(set([0, num_steps // 2, num_steps]))
    nsave = len(save_steps)

    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

    # KL modes (only if stochastic)
    if rand_R > 0:
        eigenvalues, mode_x_ids, mode_y_ids = compute_kl_modes(
            rand_R, corr_length, x_range, y_range
        )
    else:
        eigenvalues = np.zeros(0)
        mode_x_ids = np.zeros(0, dtype=int)
        mode_y_ids = np.zeros(0, dtype=int)

    def run_single_z(z_vec: np.ndarray):
        """Run full time-stepping for one Z realization. Returns snapshots list."""
        R_eff = len(z_vec)

        # Interpolate k(x,y,Z) onto mesh
        def k_interp(x_arr):
            return diffusion_coeff(
                x_arr[0],
                x_arr[1],
                z_vec,
                eigenvalues[:R_eff],
                mode_x_ids[:R_eff],
                mode_y_ids[:R_eff],
                sigma,
                mu_k,
                x_range,
                y_range,
            )

        k_func.interpolate(k_interp)

        # t=0: initial condition
        def u0_interp(x_arr):
            return manufactured_solution(
                x_arr[0],
                x_arr[1],
                0.0,
                z_vec,
                eigenvalues[:R_eff],
                mode_x_ids[:R_eff],
                mode_y_ids[:R_eff],
                gamma,
                x_range,
                y_range,
            )

        u_prev.interpolate(u0_interp)
        gD.interpolate(u0_interp)
        bc = fem.dirichletbc(value=gD, dofs=bd_dofs)

        # Bilinear form: (u/dt, v) + (k ∇u, ∇v)
        a_form = (1.0 / dt) * ufl.inner(u, v) * ufl.dx + ufl.inner(
            k_func * ufl.grad(u), ufl.grad(v)
        ) * ufl.dx

        snapshots = [u_prev.x.array.copy()]

        for n in range(1, num_steps + 1):
            t_n = n * dt

            # Update BC
            def gD_interp(x_arr, tt=t_n):
                return manufactured_solution(
                    x_arr[0],
                    x_arr[1],
                    tt,
                    z_vec,
                    eigenvalues[:R_eff],
                    mode_x_ids[:R_eff],
                    mode_y_ids[:R_eff],
                    gamma,
                    x_range,
                    y_range,
                )

            gD.interpolate(gD_interp)

            # Update source
            def src_interp(x_arr, tt=t_n):
                return source_term_darcy(
                    x_arr[0],
                    x_arr[1],
                    tt,
                    z_vec,
                    eigenvalues[:R_eff],
                    mode_x_ids[:R_eff],
                    mode_y_ids[:R_eff],
                    sigma,
                    gamma,
                    mu_k,
                    x_range,
                    y_range,
                )

            s_func.interpolate(src_interp)

            # NOTE: k doesn't change with t (spatial + stochastic only)
            # so a_form and k_func stay the same across time steps.

            L_form = (1.0 / dt) * ufl.inner(u_prev, v) * ufl.dx + ufl.inner(
                s_func, v
            ) * ufl.dx

            problem = LinearProblem(
                a_form,
                L_form,
                bcs=[bc],
                petsc_options_prefix="darcy_",
                petsc_options=petsc_opts,
            )
            u_next = problem.solve()
            u_prev.x.array[:] = u_next.x.array

            if n in save_steps:
                snapshots.append(u_prev.x.array.copy())

        return snapshots

    # ---- Deterministic mode ----
    if rand_R <= 0:
        z_zero = np.zeros(max(1, rand_R))
        snaps = run_single_z(z_zero)

        if comm.rank == 0:
            out = np.vstack(snaps[:nsave])
            np.savez(
                output,
                mode="det",
                u_ref=out,
                mesh_size=mesh_size,
                num_steps=num_steps,
                t_final=t_final,
                sigma=sigma,
                gamma=gamma,
                mu_k=mu_k,
                corr_length=corr_length,
            )
            print(f"[deterministic] saved -> {output}")
        return

    # ---- Stochastic mode: quadrature ----
    if quadrature != "tensor":
        raise NotImplementedError("Only tensor-product quadrature implemented.")

    Z_quad, W_quad = build_quadrature(rand_R, rand_p, dist=dist)
    if Z_quad.shape[0] == 0:
        raise ValueError("rand_R>0 but no quadrature nodes generated.")

    n_quad = Z_quad.shape[0]
    ndofs_local = u_prev.x.array.size

    E = np.zeros((nsave, ndofs_local), dtype=np.float64)
    E2 = np.zeros_like(E)

    print(f"[stochastic] R={rand_R}, p={rand_p}, n_quad={n_quad}, ndofs={ndofs_local}")

    for iq in range(n_quad):
        z_vec = Z_quad[iq, :]
        w = float(W_quad[iq])

        snaps = run_single_z(z_vec)

        for si, snap in enumerate(snaps[:nsave]):
            E[si] += w * snap
            E2[si] += w * (snap**2)

        if (iq + 1) % max(1, n_quad // 10) == 0:
            print(f"  quad node {iq + 1}/{n_quad} done")

    Var = E2 - E**2

    if comm.rank == 0:
        np.savez(
            output,
            mode="rand",
            E_u=E,
            Var_u=Var,
            mesh_size=mesh_size,
            num_steps=num_steps,
            t_final=t_final,
            rand_R=rand_R,
            rand_p=rand_p,
            quadrature=quadrature,
            dist=dist,
            sigma=sigma,
            gamma=gamma,
            mu_k=mu_k,
            corr_length=corr_length,
        )
        print(f"[stochastic] saved -> {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FEniCS reference solver for Darcy flow")
    ap.add_argument("--mesh", type=int, default=64)
    ap.add_argument("--t_final", type=float, default=1.0)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--rand_R", type=int, default=0)
    ap.add_argument("--rand_p", type=int, default=0)
    ap.add_argument("--quadrature", type=str, default="tensor", choices=["tensor"])
    ap.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--mu_k", type=float, default=0.0)
    ap.add_argument("--corr_length", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="darcy2d_ref.npz")
    args = ap.parse_args()

    solve_darcy2d(
        mesh_size=args.mesh,
        t_final=args.t_final,
        num_steps=args.num_steps,
        rand_R=args.rand_R,
        rand_p=args.rand_p,
        quadrature=args.quadrature,
        dist=args.dist,
        sigma=args.sigma,
        gamma=args.gamma,
        mu_k=args.mu_k,
        corr_length=args.corr_length,
        output=args.output,
    )
