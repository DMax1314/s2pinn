import numpy as np
import argparse
import os

try:
    import dolfinx
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
except ImportError:
    raise ImportError("FEniCS/dolfinx required for reference solver.")

def manufactured_solution(x, y, z, t):
    pi = np.pi
    return np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z) * np.exp(-t)

def solve_ns3d_mms(mesh_size=8, t_final=1.0, num_steps=20, nu=0.01, rand_R=0, rand_p=0, quadrature="tensor", output="ns3d_mms_ref.npz"):
    from dolfinx import mesh, fem, io
    from ufl import TrialFunction, TestFunction, dx, grad, dot
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, mesh_size, mesh_size, mesh_size)
    V = fem.FunctionSpace(domain, ("CG", 1))

    dt = t_final / num_steps
    u = fem.Function(V)
    v = TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    t = 0.0

    # Initial condition
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: manufactured_solution(x[0], x[1], x[2], 0.0))
    u.x.array[:] = u_init.x.array[:]

    # Weak form for Navier–Stokes (MMS):
    def f_rhs(x, y, z, t):
        pi = np.pi
        return manufactured_solution(x, y, z, t) * (pi ** 2 * 3 * np.exp(-t) - nu * pi ** 2 * 3 * np.exp(-t))

    results = []
    for step in range(num_steps):
        t += dt
        f = fem.Function(V)
        f.interpolate(lambda x: f_rhs(x[0], x[1], x[2], t))
        # Time stepping (simplified, treat as sequence of Poisson solves)
        a = (u * v + dt * nu * dot(grad(u), grad(v))) * dx
        L = (u_init * v + dt * f * v) * dx
        uh = fem.Function(V)
        problem = fem.petsc.LinearProblem(a, L, u=uh)
        problem.solve()
        u.x.array[:] = uh.x.array[:]
        u_init.x.array[:] = u.x.array[:]
        results.append(u.x.array.copy())

    np.savez(output, u_ref=np.array(results))
    print(f"Saved reference solution to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=8)
    parser.add_argument("--t_final", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--rand_R", type=int, default=0)
    parser.add_argument("--rand_p", type=int, default=0)
    parser.add_argument("--quadrature", type=str, default="tensor")
    parser.add_argument("--output", type=str, default="ns3d_mms_ref.npz")
    args = parser.parse_args()
    solve_ns3d_mms(
        mesh_size=args.mesh,
        t_final=args.t_final,
        num_steps=args.num_steps,
        nu=args.nu,
        rand_R=args.rand_R,
        rand_p=args.rand_p,
        quadrature=args.quadrature,
        output=args.output,
    )
