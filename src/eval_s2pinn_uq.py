"""Evaluate S2-PINN UQ calibration from an existing checkpoint."""

import argparse
import csv
import os
import sys
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))

from spinn.train import build_model, build_pde, evaluate_uq, load_config, set_seed


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint must contain model_state_dict: {path}")
    return ckpt


def _merge_eval_config(
    ckpt_cfg: Dict[str, Any],
    file_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use checkpoint architecture, with file eval defaults when needed."""
    cfg = ckpt_cfg
    if file_cfg is not None:
        cfg.setdefault("eval", file_cfg.get("eval", {}))
        cfg.setdefault("pde", file_cfg.get("pde", {}))
    cfg.setdefault("eval", {}).setdefault("uq", {})
    cfg["eval"]["uq"]["enabled"] = True
    return cfg


def _write_csv(metrics: Dict[str, float], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate UQ calibration metrics for a trained S2-PINN checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--pde",
        type=str,
        required=True,
        choices=["diffusion", "allen_cahn", "burgers", "darcy"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--n_probe", type=int, default=2048)
    parser.add_argument("--n_z_moment", type=int, default=1024)
    parser.add_argument("--n_z_interval", type=int, default=1024)
    parser.add_argument("--n_z_test", type=int, default=1024)
    parser.add_argument("--batch_limit", type=int, default=262144)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ckpt = _load_checkpoint(args.checkpoint, device)
    file_cfg = load_config(args.config) if args.config is not None else None
    ckpt_cfg = ckpt.get("config")
    if not isinstance(ckpt_cfg, dict):
        if file_cfg is None:
            raise ValueError("Need --config when checkpoint does not contain config")
        ckpt_cfg = file_cfg

    cfg = _merge_eval_config(ckpt_cfg, file_cfg)
    cfg["pde"]["type"] = args.pde
    r_dim = int(cfg["model"]["stochastic"]["R"])
    cfg["pde"]["kl"]["n_terms"] = r_dim

    uq_cfg = cfg["eval"]["uq"]
    uq_cfg["enabled"] = True
    uq_cfg["n_probe"] = int(args.n_probe)
    uq_cfg["n_z_moment"] = int(args.n_z_moment)
    uq_cfg["n_z_interval"] = int(args.n_z_interval)
    uq_cfg["n_z_test"] = int(args.n_z_test)
    uq_cfg["batch_limit"] = int(args.batch_limit)

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pde = build_pde(cfg)

    metrics = evaluate_uq(model, pde, uq_cfg, device, dtype)

    print("=== S2-PINN UQ Metrics ===")
    print(f"uq/pde={args.pde}")
    print(f"uq/calibration_seed={args.seed}")
    print(f"uq/checkpoint={args.checkpoint}")
    for key in sorted(metrics):
        print(f"{key}={metrics[key]:.8e}")

    _write_csv(metrics, args.csv_path)
    print(f"uq/csv_path={args.csv_path}")


if __name__ == "__main__":
    main()
