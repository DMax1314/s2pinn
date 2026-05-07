import argparse
import numpy as np
from typing import Tuple, List

try:
    from mpi4py import MPI
    import ufl
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    raise ImportError("FEniCS/dolfinx required for reference solver.") from e

# -------------------------
# Manufactured solution & source
# -------------------------
def manufactured_solution_xy_t(x: np.ndarray, y: np.ndarray, t: float, amp: float = 1.0) -> np.ndarray:
    pi = np.pi
    return amp * np.sin(pi * x) * np.sin(pi * y) * np.exp(-t)

def source_term_xy_t(x: np.ndarray, y: np.ndarray, t: float, amp: float = 1.0, kappa: float = 1.0) -> np.ndarray:
    # u_t = -u_true; Δu = -2*pi^2*u_true  => s = (-1 + 2*pi^2*kappa) u_true
    pi = np.pi
    u = manufactured_solution_xy_t(x, y, t, amp=amp)
    return (-1.0 + 2.0 * (pi ** 2) * kappa) * u

# -------------------------
# Simple tensor-product quadrature for random runs
# -------------------------
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
    Ws = np.meshgrid(*weights_1d, indexing="ij")
    Z = np.stack([g.reshape(-1) for g in grids], axis=1)
    w = np.prod([wi.reshape(-1) for wi in Ws], axis=0)
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

def amplitude_A(zrow: np.ndarray) -> float:
    if zrow.size == 0:
        return 1.0
    return 1.0 + 0.3 / max(1.0, np.sqrt(zrow.size)) * float(np.sum(zrow))

# -------------------------
# FEM reference solve (dolfinx 0.9 API)
# -------------------------
def solve_diffusion2d(
    mesh_size: int = 32,
    t_final: float = 1.0,
    num_steps: int = 50,
    rand_R: int = 0,
    rand_p: int = 0,
    quadrature: str = "tensor",
    dist: str = "normal",
    output: str = "diff2d_ref.npz",
    kappa: float = 1.0,
):
    """
    u_t - kappa * Δu = s   on Ω=(0,1)^2, Dirichlet BC = u_true
    - rand_R = 0: deterministic, save snapshots at t={0, 0.5 T, T}
    - rand_R > 0: random quadrature over A(Z), save E[u], Var[u] at same times
    """
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_size, mesh_size)

    # NOTE: dolfinx 0.9: use fem.functionspace (lowercase), not fem.FunctionSpace
    V = fem.functionspace(domain, ("Lagrange", 1))  # ("CG",1) 也行

    # boundary dofs
    def on_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    bd_dofs = fem.locate_dofs_geometrical(V, on_boundary)

    # time stepping
    dt = float(t_final) / int(num_steps)
    u_prev = fem.Function(V, name="u_prev")
    gD = fem.Function(V, name="gD")       # Dirichlet BC value
    s_func = fem.Function(V, name="s")    # source term

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = (1.0 / dt) * ufl.inner(u, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

    # we save t in {0, 0.5 T, T}
    save_steps = sorted(set([0, num_steps // 2, num_steps]))
    nsave = len(save_steps)

    # ---- deterministic
    if rand_R <= 0:
        # t=0 initial condition & boundary
        u_prev.interpolate(lambda x: manufactured_solution_xy_t(x[0], x[1], 0.0, amp=1.0))
        gD.interpolate(lambda x: manufactured_solution_xy_t(x[0], x[1], 0.0, amp=1.0))
        bc = fem.dirichletbc(value=gD, dofs=bd_dofs)

        snapshots = [u_prev.x.array.copy()]
        for n in range(1, num_steps + 1):
            t = n * dt
            gD.interpolate(lambda x, tt=t: manufactured_solution_xy_t(x[0], x[1], tt, amp=1.0))
            s_func.interpolate(lambda x, tt=t: source_term_xy_t(x[0], x[1], tt, amp=1.0, kappa=kappa))

            L_form = (1.0 / dt) * ufl.inner(u_prev, v) * ufl.dx + ufl.inner(s_func, v) * ufl.dx
            problem = LinearProblem(a_form, L_form, bcs=[bc], petsc_options=petsc_opts)
            u_next = problem.solve()
            u_prev.x.array[:] = u_next.x.array

            if n in save_steps:
                snapshots.append(u_prev.x.array.copy())

        if comm.rank == 0:
            out = np.vstack(snapshots[:nsave])
            np.savez(output, mode="det", u_ref=out, mesh_size=mesh_size, num_steps=num_steps, t_final=t_final)
            print(f"[deterministic] saved -> {output}")
        return

    # ---- random via tensor-product quadrature
    if quadrature != "tensor":
        raise NotImplementedError("Only tensor-product quadrature is implemented here.")
    Z, W = build_quadrature(rand_R, rand_p, dist=dist)
    if Z.shape[0] == 0:
        raise ValueError("rand_R>0 but no quadrature nodes were generated.")

    ndofs_local = u_prev.x.array.size
    E = np.zeros((nsave, ndofs_local), dtype=np.float64)
    E2 = np.zeros_like(E)

    for i in range(Z.shape[0]):
        z = Z[i, :]
        w = float(W[i])
        amp = amplitude_A(z)

        # t=0
        u_prev.interpolate(lambda x, aa=amp: manufactured_solution_xy_t(x[0], x[1], 0.0, amp=aa))
        gD.interpolate(lambda x, aa=amp: manufactured_solution_xy_t(x[0], x[1], 0.0, amp=aa))
        bc = fem.dirichletbc(value=gD, dofs=bd_dofs)

        idx = 0
        E[idx]  += w * u_prev.x.array
        E2[idx] += w * (u_prev.x.array ** 2)

        for n in range(1, num_steps + 1):
            t = n * dt
            gD.interpolate(lambda x, tt=t, aa=amp: manufactured_solution_xy_t(x[0], x[1], tt, amp=aa))
            s_func.interpolate(lambda x, tt=t, aa=amp: source_term_xy_t(x[0], x[1], tt, amp=aa, kappa=kappa))

            L_form = (1.0 / dt) * ufl.inner(u_prev, v) * ufl.dx + ufl.inner(s_func, v) * ufl.dx
            problem = LinearProblem(a_form, L_form, bcs=[bc], petsc_options=petsc_opts)
            u_next = problem.solve()
            u_prev.x.array[:] = u_next.x.array

            if n in save_steps:
                idx += 1
                E[idx]  += w * u_prev.x.array
                E2[idx] += w * (u_prev.x.array ** 2)

    Var = E2 - E ** 2
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
        )
        print(f"[random] saved -> {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=32)
    ap.add_argument("--t_final", type=float, default=1.0)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--rand_R", type=int, default=0)
    ap.add_argument("--rand_p", type=int, default=0)
    ap.add_argument("--quadrature", type=str, default="tensor", choices=["tensor"])
    ap.add_argument("--dist", type=str, default="normal", choices=["normal","uniform"])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="diff2d_ref.npz")
    args = ap.parse_args()

    solve_diffusion2d(
        mesh_size=args.mesh,
        t_final=args.t_final,
        num_steps=args.num_steps,
        rand_R=args.rand_R,
        rand_p=args.rand_p,
        quadrature=args.quadrature,
        dist=args.dist,
        output=args.output,
        kappa=args.kappa,
    )
