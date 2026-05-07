import argparse
import csv
import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(__file__))

_TRAIN_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "train_baseline.py")
_TRAIN_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "train_baseline_module", _TRAIN_BASELINE_PATH
)
if _TRAIN_BASELINE_SPEC is None or _TRAIN_BASELINE_SPEC.loader is None:
    raise ImportError(f"Unable to load train_baseline module at {_TRAIN_BASELINE_PATH}")
_train_baseline = importlib.util.module_from_spec(_TRAIN_BASELINE_SPEC)
_TRAIN_BASELINE_SPEC.loader.exec_module(_train_baseline)

_rel_l2 = _train_baseline._rel_l2
build_pde = _train_baseline.build_pde
build_vanilla_model = _train_baseline.build_vanilla_model
load_config = _train_baseline.load_config
set_seed = _train_baseline.set_seed


def _build_checkpoint_paths(args: argparse.Namespace) -> List[str]:
    if args.ckpt_paths:
        return args.ckpt_paths

    if len(args.seeds) < args.n_members:
        raise ValueError(
            f"Need at least {args.n_members} seeds, got {len(args.seeds)}."
        )

    paths: List[str] = []
    for seed in args.seeds[: args.n_members]:
        fname = (
            f"vanilla_{args.pde}_h{args.hidden_dim}_L{args.n_layers}"
            f"_drop{args.dropout}_seed{seed}.pt"
        )
        paths.append(os.path.join(args.ckpt_dir, fname))
    return paths


@torch.no_grad()
def _predict_batched(
    model: torch.nn.Module,
    t_rep: torch.Tensor,
    x_rep: torch.Tensor,
    y_rep: torch.Tensor,
    z_rep: torch.Tensor,
    batch_limit: int,
) -> torch.Tensor:
    total = t_rep.shape[0]
    preds: List[torch.Tensor] = []
    for start in range(0, total, batch_limit):
        end = min(start + batch_limit, total)
        preds.append(
            model(
                t_rep[start:end], x_rep[start:end], y_rep[start:end], z_rep[start:end]
            )
        )
    return torch.cat(preds, dim=0)


@torch.no_grad()
def _exact_batched(
    pde: Any,
    t_rep: torch.Tensor,
    x_rep: torch.Tensor,
    y_rep: torch.Tensor,
    z_rep: torch.Tensor,
    batch_limit: int,
) -> torch.Tensor:
    total = t_rep.shape[0]
    refs: List[torch.Tensor] = []
    for start in range(0, total, batch_limit):
        end = min(start + batch_limit, total)
        refs.append(
            pde.exact_solution(
                t_rep[start:end], x_rep[start:end], y_rep[start:end], z_rep[start:end]
            )
        )
    return torch.cat(refs, dim=0)


def _expand_inputs(
    t: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    n_probe = t.shape[0]
    n_z = z_samples.shape[0]
    t_rep = t.unsqueeze(1).expand(n_probe, n_z).reshape(-1)
    x_rep = x.unsqueeze(1).expand(n_probe, n_z).reshape(-1)
    y_rep = y.unsqueeze(1).expand(n_probe, n_z).reshape(-1)
    z_rep = (
        z_samples.unsqueeze(0).expand(n_probe, n_z, -1).reshape(-1, z_samples.shape[1])
    )
    return t_rep, x_rep, y_rep, z_rep, n_probe, n_z


@torch.no_grad()
def evaluate_ensemble(
    models: List[torch.nn.Module],
    pde: Any,
    device: torch.device,
    dtype: torch.dtype,
    n_probe: int,
    n_z_moment: int,
    n_z_interval: int,
    n_z_test: int,
    batch_limit: int,
) -> Dict[str, float]:
    probes = pde.sample_collocation(n_probe, device=device, dtype=dtype)
    t = probes["t"].squeeze(-1)
    x = probes["x"].squeeze(-1)
    y = probes["y"].squeeze(-1)

    z_moment = pde.sample_z(n_z_moment, device=device, dtype=dtype)
    t_m, x_m, y_m, z_m, n_probe_m, n_z_m = _expand_inputs(t, x, y, z_moment)

    moment_preds = []
    for model in models:
        pred_flat = _predict_batched(
            model,
            t_m,
            x_m,
            y_m,
            z_m,
            batch_limit,
        )
        moment_preds.append(pred_flat.reshape(n_probe_m, n_z_m))

    pooled_moment = torch.stack(moment_preds, dim=1).reshape(
        n_probe_m, len(models) * n_z_m
    )

    u_ref_flat = _exact_batched(
        pde,
        t_m,
        x_m,
        y_m,
        z_m,
        batch_limit,
    )
    u_ref = u_ref_flat.reshape(n_probe_m, n_z_m)

    mean_pred = pooled_moment.mean(dim=1)
    var_pred = pooled_moment.var(dim=1, unbiased=False)
    mean_ref = u_ref.mean(dim=1)
    var_ref = u_ref.var(dim=1, unbiased=False)

    metrics: Dict[str, float] = {
        "ensemble_uq/mean_pred": mean_pred.mean().item(),
        "ensemble_uq/var_pred": var_pred.mean().item(),
        "ensemble_uq/mean_rel_l2": _rel_l2(mean_pred, mean_ref).item(),
        "ensemble_uq/var_rel_l2": _rel_l2(var_pred, var_ref).item(),
    }

    z_interval = pde.sample_z(n_z_interval, device=device, dtype=dtype)
    t_i, x_i, y_i, z_i, n_probe_i, n_z_i = _expand_inputs(t, x, y, z_interval)

    interval_preds = []
    for model in models:
        pred_flat = _predict_batched(
            model,
            t_i,
            x_i,
            y_i,
            z_i,
            batch_limit,
        )
        interval_preds.append(pred_flat.reshape(n_probe_i, n_z_i))
    pooled_interval = torch.stack(interval_preds, dim=1).reshape(
        n_probe_i, len(models) * n_z_i
    )

    z_test = pde.sample_z(n_z_test, device=device, dtype=dtype)
    t_t, x_t, y_t, z_t, n_probe_t, n_z_t = _expand_inputs(t, x, y, z_test)
    u_test = _exact_batched(
        pde,
        t_t,
        x_t,
        y_t,
        z_t,
        batch_limit,
    ).reshape(n_probe_t, n_z_t)

    levels = [0.5, 0.8, 0.9, 0.95]
    cal_error = 0.0
    sharpness_90 = 0.0
    for level in levels:
        alpha = 1.0 - level
        lo_q = alpha / 2.0
        hi_q = 1.0 - alpha / 2.0
        lo = torch.quantile(pooled_interval, lo_q, dim=1)
        hi = torch.quantile(pooled_interval, hi_q, dim=1)

        inside = (u_test >= lo.unsqueeze(1)) & (u_test <= hi.unsqueeze(1))
        coverage = inside.float().mean().item()
        width = (hi - lo).mean().item()
        level_int = int(level * 100)

        metrics[f"ensemble_uq/coverage_{level_int}"] = coverage
        metrics[f"ensemble_uq/sharpness_{level_int}"] = width
        cal_error += abs(level - coverage)

        if level_int == 90:
            sharpness_90 = width

    metrics["ensemble_uq/calibration_error"] = cal_error / len(levels)
    metrics["ensemble_uq/sharpness"] = sharpness_90
    return metrics


def _save_metrics_csv(metrics: Dict[str, float], csv_path: str) -> None:
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deep-ensemble vanilla PINN UQ."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/ckpt")
    parser.add_argument("--ckpt_paths", type=str, nargs="*", default=None)
    parser.add_argument(
        "--pde",
        type=str,
        required=True,
        choices=["diffusion", "allen_cahn", "burgers", "darcy"],
    )
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_probe", type=int, default=2048)
    parser.add_argument("--n_z_moment", type=int, default=1024)
    parser.add_argument("--n_z_interval", type=int, default=1024)
    parser.add_argument("--n_z_test", type=int, default=1024)
    parser.add_argument("--batch_limit", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    cfg = load_config(args.config)
    cfg["pde"]["type"] = args.pde

    ckpt_paths = _build_checkpoint_paths(args)
    for path in ckpt_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    r_dim: int = cfg["model"]["stochastic"]["R"]
    models: List[torch.nn.Module] = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device)
        ckpt_cfg = ckpt.get("config", cfg)
        r_dim = ckpt_cfg["model"]["stochastic"]["R"]
        hidden_dim = ckpt_cfg.get("baseline", {}).get("hidden_dim", args.hidden_dim)
        n_layers = ckpt_cfg.get("baseline", {}).get("n_layers", args.n_layers)
        activation = ckpt_cfg.get("baseline", {}).get("activation", "tanh")
        dropout = float(ckpt_cfg.get("baseline", {}).get("dropout", args.dropout))

        model = build_vanilla_model(
            R=r_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
            device=device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    cfg["model"]["stochastic"]["R"] = r_dim
    cfg["pde"]["kl"]["n_terms"] = r_dim
    pde = build_pde(cfg)
    metrics = evaluate_ensemble(
        models=models,
        pde=pde,
        device=device,
        dtype=dtype,
        n_probe=args.n_probe,
        n_z_moment=args.n_z_moment,
        n_z_interval=args.n_z_interval,
        n_z_test=args.n_z_test,
        batch_limit=args.batch_limit,
    )

    print("=== Deep Ensemble UQ Metrics ===")
    print(f"ensemble_uq/n_members={len(models)}")
    print(f"ensemble_uq/pde={args.pde}")
    for key in sorted(metrics):
        print(f"{key}={metrics[key]:.8e}")

    if args.csv_path is not None:
        _save_metrics_csv(metrics, args.csv_path)
        print(f"ensemble_uq/csv_path={args.csv_path}")


if __name__ == "__main__":
    main()
