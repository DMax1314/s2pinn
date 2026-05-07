# solvers/fenics/helmholtz3d_ref.py
import argparse
import os
from typing import List, Tuple
import numpy as np

try:
    from mpi4py import MPI
    import ufl
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    raise ImportError("FEniCS/dolfinx 0.9+ is required for this reference solver.") from e

pi = np.pi

# ---------- utils ----------
def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def manufactured_u(x: np.ndarray, y: np.ndarray, z: np.ndarray, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)

def source_term(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: float, amp: float = 1.0) -> np.ndarray:
    # s = -Δu - k^2 u ; Δu = -(pi^2*3) u  => s = (3*pi^2 - k^2)*u
    return (3.0 * (pi**2) - k**2) * manufactured_u(x, y, z, amp=amp)

def hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)          # weight exp(-x^2)
    z = x / np.sqrt(2.0)         # std normal variable
    w_prob = w / np.sqrt(np.pi)  # normalize to expectation
    return z, w_prob

def legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)           # integrates on [-1,1]
    return x, 0.5 * w            # Uniform[-1,1] has pdf=1/2

def tensorize(nodes_1d: List[np.ndarray], weights_1d: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(*nodes_1d, indexing="ij")
    Ws = np.meshgrid(*weights_1d, indexing="ij")
    Z = np.stack([g.reshape(-1) for g in grids], axis=1)   # [N,R]
    w = np.prod([wi.reshape(-1) for wi in Ws], axis=0)     # [N]
    return Z, w

def build_quadrature(R: int, p: int, dist: str = "normal") -> Tuple[np.ndarray, np.ndarray]:
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

# ---------- main solver ----------
def solve_helmholtz3d(
    nx: int = 32, ny: int = 32, nz: int = 32,
    k: float = 4.0,
    rand_R: int = 0, rand_p: int = 0, dist: str = "normal",
    quadrature: str = "tensor",
    ksp_type: str = "preonly", pc_type: str = "lu",
    output: str = "helmholtz3d_ref.npz"
):
    """
    Steady Helmholtz on Ω=(0,1)^3:
      -Δu - k^2 u = s,  with Dirichlet BC u=g = u_true on ∂Ω.
    u_true = A * sin(pi x) sin(pi y) sin(pi z)
    s     = (3*pi^2 - k^2) * u_true

    Deterministic (rand_R=0): save u_ref (single snapshot).
    Random (rand_R>0): tensor-product quadrature over A(Z), save E[u], Var[u].
    """
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, nx, ny, nz)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # boundary dofs
    def on_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) |
                np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0))
    bd_dofs = fem.locate_dofs_geometrical(V, on_boundary)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx

    # PETSc solver options: default to a direct solve (robust for indefinite Helmholtz).
    petsc_opts = {"ksp_type": ksp_type, "pc_type": pc_type}

    gD = fem.Function(V, name="gD")    # Dirichlet value
    sV = fem.Function(V, name="s")     # source term

    # ---------- deterministic ----------
    if rand_R <= 0:
        gD.interpolate(lambda x: manufactured_u(x[0], x[1], x[2], amp=1.0))
        sV.interpolate(lambda x: source_term(x[0], x[1], x[2], k=k, amp=1.0))
        bc = fem.dirichletbc(value=gD, dofs=bd_dofs)

        L_form = ufl.inner(sV, v) * ufl.dx
        problem = LinearProblem(a_form, L_form, bcs=[bc], petsc_options=petsc_opts)
        u_sol = problem.solve()

        if comm.rank == 0:
            ensure_parent_dir(output)
            np.savez(output,
                     mode="det",
                     u_ref=u_sol.x.array.copy(),
                     k=k, nx=nx, ny=ny, nz=nz)
            print(f"[deterministic] saved -> {output}")
        return

    # ---------- random expectation/variance ----------
    if quadrature != "tensor":
        raise NotImplementedError("Only tensor-product quadrature is implemented.")
    Z, W = build_quadrature(rand_R, rand_p, dist=dist)
    if Z.shape[0] == 0:
        raise ValueError("rand_R>0 but no quadrature nodes were generated.")

    ndofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    E = np.zeros(ndofs, dtype=np.float64)
    E2 = np.zeros(ndofs, dtype=np.float64)

    for i in range(Z.shape[0]):
        z = Z[i, :]
        w = float(W[i])
        amp = amplitude_A(z)

        gD.interpolate(lambda x, aa=amp: manufactured_u(x[0], x[1], x[2], amp=aa))
        sV.interpolate(lambda x, aa=amp: source_term(x[0], x[1], x[2], k=k, amp=aa))
        bc = fem.dirichletbc(value=gD, dofs=bd_dofs)

        L_form = ufl.inner(sV, v) * ufl.dx
        problem = LinearProblem(a_form, L_form, bcs=[bc], petsc_options=petsc_opts)
        u_sol = problem.solve()
        u_arr = u_sol.x.array

        E += w * u_arr
        E2 += w * (u_arr**2)

    Var = E2 - E**2
    if comm.rank == 0:
        ensure_parent_dir(output)
        np.savez(output,
                 mode="rand",
                 E_u=E, Var_u=Var,
                 k=k, nx=nx, ny=ny, nz=nz,
                 rand_R=rand_R, rand_p=rand_p, dist=dist, quadrature=quadrature)
        print(f"[random] saved E[u], Var[u] -> {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--k", type=float, default=4.0, help="wavenumber k in -Δu - k^2 u = s")
    ap.add_argument("--rand_R", type=int, default=0)
    ap.add_argument("--rand_p", type=int, default=0)
    ap.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--quadrature", type=str, default="tensor", choices=["tensor"])
    ap.add_argument("--ksp_type", type=str, default="preonly", help="PETSc ksp_type (e.g., preonly, gmres)")
    ap.add_argument("--pc_type", type=str, default="lu", help="PETSc pc_type (e.g., lu, ilu, jacobi)")
    ap.add_argument("--output", type=str, default="gt/helmholtz3d_ref.npz")
    args = ap.parse_args()

    solve_helmholtz3d(
        nx=args.nx, ny=args.ny, nz=args.nz,
        k=args.k,
        rand_R=args.rand_R, rand_p=args.rand_p, dist=args.dist,
        quadrature=args.quadrature,
        ksp_type=args.ksp_type, pc_type=args.pc_type,
        output=args.output
    )
