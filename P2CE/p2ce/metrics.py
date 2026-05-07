"""Evaluation metrics for PC^2 benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from .problems import BaseProblem, grid_coordinates, make_full_samples


Array = np.ndarray


def relative_l2(pred: Array, ref: Array) -> float:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = np.sqrt(np.sum(ref**2))
    return float(np.sqrt(np.sum((pred - ref) ** 2)) / max(denom, 1e-12))


def predict_batched(pce, samples: Array, batch_size: int) -> Array:
    samples = np.asarray(samples, dtype=float)
    outputs = []
    for start in range(0, samples.shape[0], batch_size):
        stop = min(start + batch_size, samples.shape[0])
        outputs.append(np.asarray(pce.predict(samples[start:stop])).reshape(-1))
    return np.concatenate(outputs, axis=0)


def evaluate_reduced_moments(reduced_pce, coordinates: Array) -> Tuple[Array, Array]:
    means = np.empty(coordinates.shape[0], dtype=float)
    variances = np.empty(coordinates.shape[0], dtype=float)
    for i, coord in enumerate(coordinates):
        mean, var = reduced_pce.evaluate_coordinate(np.asarray(coord, dtype=float))
        means[i] = float(np.asarray(mean).reshape(-1)[0])
        variances[i] = max(float(np.asarray(var).reshape(-1)[0]), 0.0)
    return means, variances


def diffusion_exact_moment_metrics(
    problem: BaseProblem,
    reduced_pce,
    grid_res: int,
    t_value: float,
) -> Dict[str, float]:
    coordinates = grid_coordinates(problem, grid_res=grid_res, t_value=t_value)
    mean_pred, var_pred = evaluate_reduced_moments(reduced_pce, coordinates)
    mean_ref, var_ref = problem.exact_moments(coordinates)
    return {
        "mean_rel_l2": relative_l2(mean_pred, mean_ref),
        "var_rel_l2": relative_l2(var_pred, var_ref),
        "mean_abs_max": float(np.max(np.abs(mean_pred - mean_ref))),
        "var_abs_max": float(np.max(np.abs(var_pred - var_ref))),
    }


def load_reference_tensor(path: str | Path) -> Array:
    """Load an existing S2-PINN FDM reference tensor saved with torch.save."""
    import torch

    tensor = torch.load(str(Path(path).expanduser()), map_location="cpu")
    return np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor)


def reference_moment_metrics(
    problem: BaseProblem,
    reduced_pce,
    grid_res: int,
    mean_ref_path: str | Path,
    var_ref_path: str | Path,
) -> Dict[str, float]:
    coordinates = grid_coordinates(problem, grid_res=grid_res)
    mean_pred, var_pred = evaluate_reduced_moments(reduced_pce, coordinates)
    mean_ref = load_reference_tensor(mean_ref_path).reshape(-1)
    var_ref = load_reference_tensor(var_ref_path).reshape(-1)
    return {
        "mean_rel_l2": relative_l2(mean_pred, mean_ref),
        "var_rel_l2": relative_l2(var_pred, var_ref),
        "mean_abs_max": float(np.max(np.abs(mean_pred - mean_ref))),
        "var_abs_max": float(np.max(np.abs(var_pred - var_ref))),
    }


def calibration_metrics(
    problem: BaseProblem,
    pce,
    rng: np.random.Generator,
    n_probe: int,
    n_z: int,
    levels: Iterable[float],
    batch_size: int,
) -> Dict[str, float]:
    if not problem.has_exact_solution:
        return {}

    deterministic = problem.sample_interior(rng, n_probe)[:, : problem.deterministic_dim]
    z = problem.sample_z(rng, n_z)
    levels = list(levels)
    inside_counts = {level: 0 for level in levels}
    total = 0
    sharpness_widths = []

    for coord in deterministic:
        samples = make_full_samples(problem, coord, z)
        pred = predict_batched(pce, samples, batch_size=batch_size)
        truth = problem.exact_solution(samples)
        for level in levels:
            alpha = 1.0 - level
            lower = np.quantile(pred, alpha / 2.0)
            upper = np.quantile(pred, 1.0 - alpha / 2.0)
            inside_counts[level] += int(np.count_nonzero((truth >= lower) & (truth <= upper)))
            if abs(level - 0.9) < 1e-12:
                sharpness_widths.append(float(upper - lower))
        total += truth.shape[0]

    metrics: Dict[str, float] = {}
    abs_errors = []
    for level in levels:
        coverage = inside_counts[level] / max(total, 1)
        metrics[f"coverage_{int(round(level * 100))}"] = float(coverage)
        abs_errors.append(abs(coverage - level))
    metrics["calibration_error"] = float(np.mean(abs_errors))
    metrics["sharpness_90"] = float(np.mean(sharpness_widths)) if sharpness_widths else float("nan")
    return metrics
