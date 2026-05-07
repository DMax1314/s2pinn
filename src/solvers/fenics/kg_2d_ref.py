# solvers/fenics/kg_2d_ref.py
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

def manufactured_u(x: np.ndarray, y: np.ndarray, t: float, amp: float, omega: float) -> np.ndarray:
    return amp * np.sin(pi * x) * np.sin(pi * y) * np.cos(omega * t)

def manufactured_ut(x: np.ndarray, y: np.ndarray, t: float, amp: float, omega: float) -> np.ndarray:
    # time derivative
    return -amp * omega * np.sin(pi * x) * np.sin(pi * y) * np.sin(omega * t)

def manufactured_utt(x: np.ndarray, y: np.ndarray, t: float, amp: float, omega: float) -> np.ndarray:
    # second time derivative
    return - (omega ** 2) * manufactured_u(x, y, t, amp, omega)

def source_term(x: np.ndarray, y: np.ndarray, t: float, amp: float, omega: float, m: float) -> np.ndarray:
    # s = u_tt - Δu + m^2 u ; Δu = -2*pi^2*u  for sin(pi x) sin(pi y)
    u = manufactured_u(x, y, t, amp, omega)
    utt = manufactured_utt(x, y, t, amp, omega)
    return utt + 2.0 * (pi ** 2) * u + (m ** 2) * u

def hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    z = x / np.sqrt(2.0)
    w_prob = w / np.sqrt(np.pi)
    return z, w_prob

def legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x, 0.5 * w

def tensorize(nodes_1d: List[np.ndarray], weights_1d: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(*nodes_1d, indexing="ij")
    Ws = np.meshgrid(*weights_1d, indexing="ij")
    Z = np.stack([g.reshape(-1) for g in grids], axis=1)
    w = np.prod([wi.reshape(-1) for wi in Ws], axis=0)
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

# ---------- Newmark-beta KG solver ----------
def solve_kg_2d(
    mesh_size: int = 64,
    t_final: float = 1.0,
    num_steps: int = 50,
    m: float = 1.0,
    omega: float | None = None,     # if None -> auto: sqrt(2*pi^2 + m^2) (zero source)
    rand_R: int = 0,
    rand_p: int = 0,
    dist: str = "normal",
    quadrature: str = "tensor",
    output: str = "kg_2d_ref.npz",
    beta: float = 0.25,
    gamma: float = 0.5,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
):
    """
    Klein–Gordon in (0,1)^2 with homogeneous Dirichlet BC:
      u_tt - Δu + m^2 u = s,   u|_{∂Ω}=0.

    Manufactured truth:
      u*(x,y,t;A)=A sin(pi x) sin(pi y) cos(omega t)
      s = u_tt - Δu + m^2 u
      If omega = sqrt(2*pi^2 + m^2) => s ≡ 0.

    Deterministic (rand_R=0): save snapshots at t={0, 0.5T, T}.
    Random (rand_R>0): tensor-product quadrature over A(Z), save E[u], Var[u] at same times.
    """
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_size, mesh_size)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # time grid
    dt = float(t_final) / int(num_steps)
    times = np.array([0.0, 0.5 * t_final, t_final], dtype=float)
    save_steps = sorted(set([0, num_steps // 2, num_steps]))
    nsave = len(save_steps)

    # omega auto to make s=0
    if omega is None:
        omega = float(np.sqrt(2.0 * (pi ** 2) + m ** 2))

    # boundary dofs (all sides -> homogeneous Dirichlet u=0)
    def on_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    bd_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    zero = fem.Function(V); zero.x.array[:] = 0.0
    bc = fem.dirichletbc(value=zero, dofs=bd_dofs)

    # trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # operators:
    # M(u,v) = <u,v>,  K(u,v) = <grad u, grad v> + m^2 <u,v>
    mass = ufl.inner(u, v) * ufl.dx
    stiff = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + (m ** 2) * ufl.inner(u, v) * ufl.dx

    # Newmark: solve for a_{n+1} with A = M + beta*dt^2*K~
    a_form = mass + beta * (dt ** 2) * stiff

    # source holder
    sV = fem.Function(V, name="s")

    # state variables (Functions)
    u_prev = fem.Function(V, name="u_prev")
    v_prev = fem.Function(V, name="v_prev")
    a_prev = fem.Function(V, name="a_prev")

    # init at t=0 with A=1 (deterministic) or per-node amp in random loop
    def init_state(amp: float):
        u_prev.interpolate(lambda x, aa=amp: manufactured_u(x[0], x[1], 0.0, aa, omega))
        v_prev.interpolate(lambda x, aa=amp: manufactured_ut(x[0], x[1], 0.0, aa, omega))
        a_prev.interpolate(lambda x, aa=amp: manufactured_utt(x[0], x[1], 0.0, aa, omega))

    # deterministic branch
    if rand_R <= 0:
        init_state(1.0)
        # t=0 snapshot
        snaps = [u_prev.x.array.copy()]
        for n in range(1, num_steps + 1):
            tnp1 = n * dt
            # predictors
            u_pred = fem.Function(V); u_pred.x.array[:] = (
                u_prev.x.array + dt * v_prev.x.array + (0.5 - beta) * (dt ** 2) * a_prev.x.array
            )
            v_pred = fem.Function(V); v_pred.x.array[:] = (
                v_prev.x.array + (1.0 - gamma) * dt * a_prev.x.array
            )
            # build RHS: (s_{n+1}, v) - (grad u_pred, grad v) - m^2 (u_pred, v)
            sV.interpolate(lambda x, tt=tnp1: source_term(x[0], x[1], tt, amp=1.0, omega=omega, m=m))
            L_form = ufl.inner(sV, v) * ufl.dx \
                   - ufl.inner(ufl.grad(u_pred), ufl.grad(v)) * ufl.dx \
                   - (m ** 2) * ufl.inner(u_pred, v) * ufl.dx
            # solve for a_{n+1}
            problem_acc = LinearProblem(a_form, L_form, bcs=[bc],
                                        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type})
            a_next = problem_acc.solve()
            # correctors
            u_next = fem.Function(V); u_next.x.array[:] = u_pred.x.array + beta * (dt ** 2) * a_next.x.array
            v_next = fem.Function(V); v_next.x.array[:] = v_pred.x.array + gamma * dt * a_next.x.array

            # roll
            u_prev.x.array[:] = u_next.x.array
            v_prev.x.array[:] = v_next.x.array
            a_prev.x.array[:] = a_next.x.array

            if n in save_steps:
                snaps.append(u_prev.x.array.copy())

        if MPI.COMM_WORLD.rank == 0:
            ensure_parent_dir(output)
            out = np.vstack(snaps[:nsave])
            np.savez(output, mode="det", times=times, u_ref=out,
                     mesh_size=mesh_size, num_steps=num_steps, t_final=t_final, m=m, omega=omega)
            print(f"[KG-2D det] saved -> {output}")
        return

    # random branch
    if quadrature != "tensor":
        raise NotImplementedError("Only tensor-product quadrature is implemented.")
    Z, W = build_quadrature(rand_R, rand_p, dist=dist)
    if Z.shape[0] == 0:
        raise ValueError("rand_R>0 but no quadrature nodes were generated.")

    ndofs = u_prev.x.array.size
    E = np.zeros((nsave, ndofs), dtype=np.float64)
    E2 = np.zeros_like(E)

    for i in range(Z.shape[0]):
        z = Z[i, :]
        w = float(W[i])
        amp = amplitude_A(z)

        init_state(amp)

        # save t=0
        idx = 0
        E[idx]  += w * u_prev.x.array
        E2[idx] += w * (u_prev.x.array ** 2)

        for n in range(1, num_steps + 1):
            tnp1 = n * dt
            u_pred = fem.Function(V); u_pred.x.array[:] = (
                u_prev.x.array + dt * v_prev.x.array + (0.5 - beta) * (dt ** 2) * a_prev.x.array
            )
            v_pred = fem.Function(V); v_pred.x.array[:] = (
                v_prev.x.array + (1.0 - gamma) * dt * a_prev.x.array
            )

            sV.interpolate(lambda x, tt=tnp1, aa=amp: source_term(x[0], x[1], tt, amp=aa, omega=omega, m=m))
            L_form = ufl.inner(sV, v) * ufl.dx \
                   - ufl.inner(ufl.grad(u_pred), ufl.grad(v)) * ufl.dx \
                   - (m ** 2) * ufl.inner(u_pred, v) * ufl.dx

            problem_acc = LinearProblem(a_form, L_form, bcs=[bc],
                                        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type})
            a_next = problem_acc.solve()
            u_next = fem.Function(V); u_next.x.array[:] = u_pred.x.array + beta * (dt ** 2) * a_next.x.array
            v_next = fem.Function(V); v_next.x.array[:] = v_pred.x.array + gamma * dt * a_next.x.array

            u_prev.x.array[:] = u_next.x.array
            v_prev.x.array[:] = v_next.x.array
            a_prev.x.array[:] = a_next.x.array

            if n in save_steps:
                idx += 1
                E[idx]  += w * u_prev.x.array
                E2[idx] += w * (u_prev.x.array ** 2)

    Var = E2 - E ** 2
    if MPI.COMM_WORLD.rank == 0:
        ensure_parent_dir(output)
        np.savez(output, mode="rand", times=times, E_u=E, Var_u=Var,
                 mesh_size=mesh_size, num_steps=num_steps, t_final=t_final,
                 m=m, omega=omega, rand_R=rand_R, rand_p=rand_p, dist=dist, quadrature=quadrature)
        print(f"[KG-2D rand] saved E[u], Var[u] -> {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=64)
    ap.add_argument("--t_final", type=float, default=1.0)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=float("nan"),
                    help="temporal frequency; if NaN -> auto sqrt(2*pi^2+m^2)")
    ap.add_argument("--rand_R", type=int, default=0)
    ap.add_argument("--rand_p", type=int, default=0)
    ap.add_argument("--dist", type=str, default="normal", choices=["normal","uniform"])
    ap.add_argument("--quadrature", type=str, default="tensor", choices=["tensor"])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--ksp_type", type=str, default="cg")
    ap.add_argument("--pc_type", type=str, default="hypre")
    ap.add_argument("--output", type=str, default="gt/kg_2d_ref.npz")
    args = ap.parse_args()

    omega_val = None if np.isnan(args.omega) else float(args.omega)

    solve_kg_2d(mesh_size=args.mesh,
                t_final=args.t_final,
                num_steps=args.num_steps,
                m=args.m,
                omega=omega_val,
                rand_R=args.rand_R,
                rand_p=args.rand_p,
                dist=args.dist,
                quadrature=args.quadrature,
                output=args.output,
                beta=args.beta,
                gamma=args.gamma,
                ksp_type=args.ksp_type,
                pc_type=args.pc_type)
