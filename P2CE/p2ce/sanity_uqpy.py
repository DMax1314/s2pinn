"""Remote sanity checks for UQpy PC^2 before full arrays.

This script is intended to run only on Sol through SLURM. It checks the UQpy
installation, runs a compact version of UQpy's documented wave-equation PC^2
example, and fits tiny versions of the target PDEs without evaluating test
metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .runner import build_pc2, fit_pc2, load_config


def run_wave_example() -> float:
    from UQpy.distributions import JointIndependent, Uniform
    from UQpy.surrogates import LeastSquareRegression, PolynomialChaosExpansion, TotalDegreeBasis
    from UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE import (
        ConstrainedPCE,
    )
    from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
    from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import PdePCE
    from UQpy.surrogates.polynomial_chaos.physics_informed.Utilities import (
        derivative_basis,
        transformation_multiplier,
    )

    rng = np.random.default_rng(2026)
    joint = JointIndependent(marginals=[Uniform(loc=0, scale=1), Uniform(loc=0, scale=2)])
    pde_data = PdeData(
        [1, 2],
        [0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [
            np.column_stack(
                [
                    np.r_[np.zeros(8), np.ones(8)],
                    np.r_[np.linspace(0, 2, 8), np.linspace(0, 2, 8)],
                ]
            ),
            np.column_stack([np.linspace(0, 1, 8), np.zeros(8)]),
            np.column_stack([np.linspace(0, 1, 8), np.zeros(8)]),
        ],
        [np.zeros(16), np.zeros(8), np.sin(np.pi * np.linspace(0, 1, 8))],
    )

    def pde_basis(sample, pce):
        dxx = derivative_basis(sample, pce, derivative_order=2, leading_variable=0)
        dtt = derivative_basis(sample, pce, derivative_order=2, leading_variable=1)
        dxx *= transformation_multiplier(pde_data, leading_variable=0, derivation_order=2)
        dtt *= transformation_multiplier(pde_data, leading_variable=1, derivation_order=2)
        return dtt - 4.0 * dxx

    def pde_source(sample):
        return np.zeros(sample.shape[0])

    def virtual_sampling(nsim):
        x = rng.uniform(0.0, 1.0, size=(nsim, 1))
        t = rng.uniform(0.0, 2.0, size=(nsim, 1))
        return np.hstack([x, t])

    def bc_error(nsim, pce):
        x = virtual_sampling(nsim)
        x[: nsim // 2, 0] = 0.0
        x[nsim // 2 :, 0] = 1.0
        return np.asarray(pce.predict(x)).reshape(-1)

    dirichlet = pde_data.dirichlet
    pce = PolynomialChaosExpansion(
        polynomial_basis=TotalDegreeBasis(joint, 4),
        regression_method=LeastSquareRegression(),
    )
    pce.set_data(dirichlet[:, :-1], dirichlet[:, -1])
    pde_pce = PdePCE(
        pde_data,
        pde_basis,
        pde_source=pde_source,
        boundary_conditions_evaluate=bc_error,
        virtual_points_sampling=virtual_sampling,
    )
    pcpc = ConstrainedPCE(pde_data, pde_pce, pce)
    pcpc.ols(nvirtual=32, n_error_points=64)

    eval_x = virtual_sampling(128)
    pred = np.asarray(pcpc.initial_pce.predict(eval_x)).reshape(-1)
    truth = np.sin(np.pi * eval_x[:, 0]) * np.cos(2.0 * np.pi * eval_x[:, 1])
    mse = float(np.mean((pred - truth) ** 2))
    print(f"[sanity] documented wave mini-example MSE={mse:.6e}")
    return mse


def run_target_tiny(config_path: Path) -> None:
    raw = load_config(config_path)
    raw["problem"]["R"] = 1
    raw["pc2"]["p"] = 2
    raw["pc2"]["solver"] = "ols"
    raw["pc2"]["n_boundary"] = 16
    raw["pc2"]["n_initial"] = 16
    raw["pc2"]["n_virtual"] = 32
    raw["pc2"]["n_error_points"] = 64
    raw["pc2"]["n_validation"] = 64
    raw["output"]["save_model"] = False
    rng = np.random.default_rng(int(raw["seed"]))
    problem, _, _, pcpc = build_pc2(raw, rng)
    _, elapsed, fit_error = fit_pc2(raw, pcpc)
    print(
        f"[sanity] target={problem.cfg.problem_type} R=1 p=2 "
        f"fit_error={fit_error:.6e} elapsed={elapsed:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="P2CE", help="Path to P2CE root on Sol.")
    args = parser.parse_args()

    import UQpy

    print(f"[sanity] UQpy version={getattr(UQpy, '__version__', 'unknown')}")
    run_wave_example()
    root = Path(args.root)
    for config_name in ("diffusion_r5.yaml", "poisson_mc_r5.yaml", "darcy_mc_r5.yaml"):
        run_target_tiny(root / "configs" / config_name)
    print("[sanity] all PC^2 sanity checks completed")


if __name__ == "__main__":
    main()
