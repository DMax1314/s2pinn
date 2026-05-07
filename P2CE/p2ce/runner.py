"""Cloud runner for UQpy physics-informed polynomial chaos experiments."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from .metrics import (
    calibration_metrics,
    diffusion_exact_moment_metrics,
    predict_batched,
    reference_moment_metrics,
)
from .problems import DarcyMCProblem, problem_from_dict


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_nested(raw: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = raw
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def make_distributions(problem):
    from UQpy.distributions import JointIndependent, Normal, Uniform

    distributions = []
    for low, high in zip(problem.lower_bounds, problem.upper_bounds):
        distributions.append(Uniform(loc=low, scale=high - low))
    for _ in range(problem.R):
        distributions.append(Normal(loc=0.0, scale=1.0))
    return JointIndependent(marginals=distributions)


def _total_degree_multi_index(inputs_number: int, degree: int) -> np.ndarray:
    """Return all non-negative multi-indices with total degree <= degree."""
    if inputs_number < 0:
        raise ValueError(f"inputs_number must be non-negative, got {inputs_number}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if inputs_number == 0:
        return np.zeros((1, 0), dtype=int)

    rows: list[list[int]] = []

    def recurse(prefix: list[int], remaining: int, variables_left: int) -> None:
        if variables_left == 1:
            for value in range(remaining + 1):
                rows.append(prefix + [value])
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, variables_left - 1)

    recurse([], degree, inputs_number)
    return np.asarray(rows, dtype=int)


def _anisotropic_product_multi_index(
    deterministic_dim: int,
    stochastic_dim: int,
    deterministic_degree: int,
    stochastic_degree: int,
) -> np.ndarray:
    deterministic = _total_degree_multi_index(deterministic_dim, deterministic_degree)
    stochastic = _total_degree_multi_index(stochastic_dim, stochastic_degree)
    rows = [
        np.concatenate([deterministic_row, stochastic_row])
        for deterministic_row in deterministic
        for stochastic_row in stochastic
    ]
    return np.asarray(rows, dtype=int)


def make_polynomial_basis(distributions, problem, run_cfg: Dict[str, Any]):
    basis_type = str(run_cfg.get("basis_type", "total_degree")).lower()
    if basis_type in {"total_degree", "native_total_degree"}:
        from UQpy.surrogates import TotalDegreeBasis

        return TotalDegreeBasis(distributions, int(run_cfg["p"]))

    if basis_type in {"anisotropic", "anisotropic_product", "product"}:
        from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import (
            PolynomialBasis,
        )

        deterministic_degree = int(
            run_cfg.get("p_deterministic", run_cfg.get("p_det", run_cfg.get("p", 3)))
        )
        stochastic_degree = int(
            run_cfg.get("p_stochastic", run_cfg.get("p_z", run_cfg.get("p", 3)))
        )
        multi_index = _anisotropic_product_multi_index(
            deterministic_dim=problem.deterministic_dim,
            stochastic_dim=problem.R,
            deterministic_degree=deterministic_degree,
            stochastic_degree=stochastic_degree,
        )
        polynomials = PolynomialBasis.construct_arbitrary_basis(
            problem.input_dim, distributions, multi_index
        )
        return PolynomialBasis(
            problem.input_dim,
            len(multi_index),
            multi_index,
            polynomials,
            distributions,
        )

    raise ValueError(f"Unsupported PC^2 basis_type: {basis_type}")


def _basis_type(raw: Dict[str, Any]) -> str:
    return str(raw["pc2"].get("basis_type", "total_degree")).lower()


def _p_stochastic(raw: Dict[str, Any]) -> int:
    run_cfg = raw["pc2"]
    return int(run_cfg.get("p_stochastic", run_cfg.get("p_z", run_cfg.get("p", -1))))


def _p_deterministic(raw: Dict[str, Any]) -> int:
    run_cfg = raw["pc2"]
    return int(run_cfg.get("p_deterministic", run_cfg.get("p_det", run_cfg.get("p", -1))))


def result_stem(raw: Dict[str, Any]) -> str:
    basis_type = _basis_type(raw)
    if basis_type in {"anisotropic", "anisotropic_product", "product"}:
        degree_label = f"pz{_p_stochastic(raw)}_pdet{_p_deterministic(raw)}"
        basis_label = "aniso"
    else:
        degree_label = f"p{raw['pc2']['p']}"
        basis_label = "td"
    return (
        f"pc2_{basis_label}_{raw['problem']['type']}_R{raw['problem']['R']}"
        f"_{degree_label}_{raw['pc2']['solver']}_seed{raw['seed']}"
    )


def _basis_count(pce) -> int:
    basis = getattr(pce, "polynomial_basis", None)
    if basis is None:
        return -1
    for attr in ("polynomials_number", "basis_number", "number_of_polynomials"):
        value = getattr(basis, attr, None)
        if value is not None:
            return int(value)
    polynomials = getattr(basis, "polynomials", None)
    if polynomials is not None:
        return len(polynomials)
    coefficients = getattr(pce, "coefficients", None)
    return int(np.asarray(coefficients).reshape(-1).shape[0]) if coefficients is not None else -1


def make_pde_functions(problem):
    from UQpy.surrogates.polynomial_chaos.physics_informed.Utilities import derivative_basis

    def dx_basis(standardized_sample, pce, order):
        return derivative_basis(standardized_sample, pce, order, leading_variable=0) * (
            problem.derivative_multiplier(0, order)
        )

    def dy_basis(standardized_sample, pce, order):
        return derivative_basis(standardized_sample, pce, order, leading_variable=1) * (
            problem.derivative_multiplier(1, order)
        )

    def dt_basis(standardized_sample, pce, order):
        return derivative_basis(standardized_sample, pce, order, leading_variable=2) * (
            problem.derivative_multiplier(2, order)
        )

    def pde_basis(standardized_sample, pce):
        physical = problem.standardized_to_physical(standardized_sample)
        laplacian = dx_basis(standardized_sample, pce, 2) + dy_basis(
            standardized_sample, pce, 2
        )
        if problem.cfg.problem_type == "diffusion":
            k = problem.diffusion_coefficient(physical)
            return dt_basis(standardized_sample, pce, 1) - k[:, None] * laplacian
        if problem.cfg.problem_type == "poisson_mc":
            return -laplacian
        if problem.cfg.problem_type == "darcy_mc":
            if not isinstance(problem, DarcyMCProblem):
                raise TypeError("darcy_mc problem must be a DarcyMCProblem")
            k, k_x, k_y = problem.coefficient_and_derivatives(physical)
            return (
                -k_x[:, None] * dx_basis(standardized_sample, pce, 1)
                - k_y[:, None] * dy_basis(standardized_sample, pce, 1)
                - k[:, None] * laplacian
            )
        raise ValueError(f"Unsupported problem type {problem.cfg.problem_type}")

    def pde_source(standardized_sample):
        physical = problem.standardized_to_physical(standardized_sample)
        return problem.source_term(physical)

    return pde_basis, pde_source


def build_pc2(raw: Dict[str, Any], rng: np.random.Generator):
    from UQpy.surrogates import LeastSquareRegression, PolynomialChaosExpansion
    from UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE import (
        ConstrainedPCE,
    )
    from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
    from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import (
        PdePCE,
    )

    problem = problem_from_dict(raw["problem"])
    run_cfg = raw["pc2"]
    distributions = make_distributions(problem)
    basis = make_polynomial_basis(distributions, problem, run_cfg)

    n_boundary = int(run_cfg.get("n_boundary", 2048))
    n_initial = int(run_cfg.get("n_initial", 0)) if problem.transient else 0
    bc_coords, bc_values = problem.sample_dirichlet(rng, n_boundary, n_initial)

    pde_data = PdeData(
        problem.upper_bounds,
        problem.lower_bounds,
        [0],
        [0],
        [bc_coords],
        [bc_values],
    )

    regression = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=basis, regression_method=regression)
    dirichlet = pde_data.dirichlet
    pce.set_data(dirichlet[:, :-1], dirichlet[:, -1])

    pde_basis, pde_source = make_pde_functions(problem)
    virtual_rng = np.random.default_rng(int(raw["seed"]) + 171)
    boundary_rng = np.random.default_rng(int(raw["seed"]) + 313)
    batch_size = int(run_cfg.get("batch_size", 200_000))

    def virtual_points_sampling(n_points: int):
        return problem.sample_interior(virtual_rng, n_points)

    def boundary_conditions_evaluate(nsim: int, fitted_pce):
        n_bc = max(int(nsim), 1)
        n_init = n_bc if problem.transient else 0
        coords, values = problem.sample_dirichlet(boundary_rng, n_bc, n_init)
        pred = predict_batched(fitted_pce, coords, batch_size=batch_size)
        return pred - values

    pde_pce = PdePCE(
        pde_data,
        pde_basis,
        pde_source=pde_source,
        boundary_conditions_evaluate=boundary_conditions_evaluate,
        virtual_points_sampling=virtual_points_sampling,
    )
    pcpc = ConstrainedPCE(pde_data, pde_pce, pce)
    return problem, distributions, pde_pce, pcpc


def fit_pc2(raw: Dict[str, Any], pcpc):
    solver = raw["pc2"].get("solver", "ols").lower()
    n_virtual = int(raw["pc2"].get("n_virtual", -1))
    n_error_points = int(raw["pc2"].get("n_error_points", 4096))
    start = time.perf_counter()
    if solver == "ols":
        pcpc.ols(nvirtual=n_virtual, n_error_points=n_error_points)
        fitted_pce = pcpc.initial_pce
        fit_error = float(getattr(pcpc, "ols_err", np.nan))
    elif solver == "lar":
        pcpc.lar(nvirtual=n_virtual, n_error_points=n_error_points)
        fitted_pce = pcpc.lar_pce
        fit_error = float(getattr(pcpc, "lar_error", np.nan))
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    elapsed = time.perf_counter() - start
    return fitted_pce, elapsed, fit_error


def validation_error(raw: Dict[str, Any], problem, distributions, pcpc, fitted_pce) -> float:
    from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials

    n_validation = int(raw["pc2"].get("n_validation", 4096))
    rng = np.random.default_rng(int(raw["seed"]) + 997)
    physical = problem.sample_interior(rng, n_validation)
    standardized = Polynomials.standardize_sample(physical, distributions)
    return float(pcpc.estimate_error(fitted_pce, standardized))


def make_reduced_pce(fitted_pce, problem):
    from UQpy.surrogates.polynomial_chaos.physics_informed.ReducedPCE import ReducedPCE

    return ReducedPCE(fitted_pce, problem.deterministic_dim)


def evaluate(raw: Dict[str, Any], problem, fitted_pce, reduced_pce) -> Dict[str, float]:
    eval_cfg = raw.get("evaluation", {})
    grid_res = int(eval_cfg.get("grid_res", 64))
    t_value = float(eval_cfg.get("t_value", 1.0))
    batch_size = int(raw["pc2"].get("batch_size", 200_000))

    if problem.cfg.problem_type == "diffusion":
        metrics = diffusion_exact_moment_metrics(problem, reduced_pce, grid_res, t_value)
        cal_cfg = eval_cfg.get("calibration", {})
        rng = np.random.default_rng(int(raw["seed"]) + 4441)
        metrics.update(
            calibration_metrics(
                problem=problem,
                pce=fitted_pce,
                rng=rng,
                n_probe=int(cal_cfg.get("n_probe", 512)),
                n_z=int(cal_cfg.get("n_z", 512)),
                levels=cal_cfg.get("levels", [0.5, 0.8, 0.9, 0.95]),
                batch_size=batch_size,
            )
        )
        return metrics

    refs = eval_cfg.get("references", {})
    if not refs.get("mean") or not refs.get("var"):
        raise ValueError(f"{problem.cfg.name} requires mean/var reference paths")
    return reference_moment_metrics(
        problem=problem,
        reduced_pce=reduced_pce,
        grid_res=grid_res,
        mean_ref_path=refs["mean"],
        var_ref_path=refs["var"],
    )


def log_wandb(raw: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    logging_cfg = raw.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", True)):
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] wandb import failed; continuing without WandB: {exc}")
        return

    run_name = logging_cfg.get("run_name")
    if not run_name:
        run_name = result_stem(raw)
    run = wandb.init(
        project=logging_cfg.get("project", "s2pinn"),
        entity=logging_cfg.get("entity"),
        name=run_name,
        config=raw,
        reinit=True,
    )
    run.log(metrics)
    run.finish()


def write_outputs(raw: Dict[str, Any], metrics: Dict[str, Any], fitted_pce) -> Path:
    result_dir = Path(raw["output"]["dir"]).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = result_stem(raw)
    json_path = result_dir / f"{stem}.json"
    csv_path = result_dir / f"{stem}.csv"
    model_path = result_dir / f"{stem}.pkl"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"config": raw, "metrics": metrics}, handle, indent=2, sort_keys=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    if bool(raw["output"].get("save_model", False)):
        try:
            with open(model_path, "wb") as handle:
                pickle.dump(fitted_pce, handle)
        except Exception as exc:
            print(f"[WARN] Could not pickle fitted UQpy PCE; metrics were saved: {exc}")
    return json_path


def run(raw: Dict[str, Any]) -> Dict[str, Any]:
    rng = np.random.default_rng(int(raw["seed"]))
    problem, distributions, pde_pce, pcpc = build_pc2(raw, rng)
    fitted_pce, elapsed, fit_error = fit_pc2(raw, pcpc)
    reduced_pce = make_reduced_pce(fitted_pce, problem)
    metrics: Dict[str, Any] = {
        "problem": raw["problem"]["type"],
        "R": int(raw["problem"]["R"]),
        "basis_type": _basis_type(raw),
        "p": int(raw["pc2"].get("p", _p_deterministic(raw))),
        "p_stochastic": _p_stochastic(raw),
        "p_deterministic": _p_deterministic(raw),
        "solver": raw["pc2"]["solver"],
        "seed": int(raw["seed"]),
        "basis_count": _basis_count(fitted_pce),
        "fit_error": fit_error,
        "elapsed_seconds": float(elapsed),
    }
    metrics["validation_error"] = validation_error(raw, problem, distributions, pcpc, fitted_pce)
    metrics.update(evaluate(raw, problem, fitted_pce, reduced_pce))
    log_wandb(raw, metrics)
    out_path = write_outputs(raw, metrics, fitted_pce)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved PC^2 metrics to {out_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a P2CE YAML config.")
    parser.add_argument("--p", type=int, help="Override polynomial total degree.")
    parser.add_argument("--p_det", type=int, help="Override deterministic polynomial degree.")
    parser.add_argument("--p_z", type=int, help="Override stochastic gPC degree.")
    parser.add_argument("--basis_type", help="Override basis type, e.g. anisotropic_product.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--solver", choices=["ols", "lar"], help="Override PC^2 solver.")
    parser.add_argument("--output_dir", help="Override output directory.")
    parser.add_argument("--set", action="append", default=[], help="Override dotted YAML keys, key=value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    if args.p is not None:
        raw["pc2"]["p"] = args.p
    if args.p_det is not None:
        raw["pc2"]["p_deterministic"] = args.p_det
    if args.p_z is not None:
        raw["pc2"]["p_stochastic"] = args.p_z
    if args.basis_type is not None:
        raw["pc2"]["basis_type"] = args.basis_type
    if args.seed is not None:
        raw["seed"] = args.seed
    if args.solver is not None:
        raw["pc2"]["solver"] = args.solver
    if args.output_dir:
        raw["output"]["dir"] = args.output_dir
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        update_nested(raw, key, parsed_value)
    run(raw)


if __name__ == "__main__":
    main()
