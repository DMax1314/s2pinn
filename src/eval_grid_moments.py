import argparse
import os
from typing import Any, Dict, Optional, Tuple

import torch

from spinn.train import build_model, build_pde, load_config
from train_baseline import build_vanilla_model


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt
    return {"model_state_dict": ckpt}


def _grid_xy(
    cfg: Dict[str, Any], grid_res: int, device: torch.device, dtype: torch.dtype
):
    dom = cfg["pde"]["domain"]
    x_lin = torch.linspace(
        dom["x_min"], dom["x_max"], grid_res, device=device, dtype=dtype
    )
    y_lin = torch.linspace(
        dom["y_min"], dom["y_max"], grid_res, device=device, dtype=dtype
    )
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")
    return x_grid, y_grid


def _expand_inputs(
    t_flat: torch.Tensor,
    x_flat: torch.Tensor,
    y_flat: torch.Tensor,
    z_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    b = x_flat.shape[0]
    nz = z_samples.shape[0]

    t_rep = t_flat.unsqueeze(1).expand(b, nz, 1).reshape(-1, 1)
    x_rep = x_flat.unsqueeze(1).expand(b, nz, 1).reshape(-1, 1)
    y_rep = y_flat.unsqueeze(1).expand(b, nz, 1).reshape(-1, 1)
    z_rep = z_samples.unsqueeze(0).expand(b, nz, -1).reshape(-1, z_samples.shape[1])
    return t_rep, x_rep, y_rep, z_rep, b, nz


@torch.no_grad()
def _forward_batched(
    model: Any,
    t_rep: torch.Tensor,
    x_rep: torch.Tensor,
    y_rep: torch.Tensor,
    z_rep: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    outs = []
    total = t_rep.shape[0]
    for s in range(0, total, batch):
        e = min(s + batch, total)
        outs.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], z_rep[s:e]))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _exact_batched(
    pde: Any,
    t_rep: torch.Tensor,
    x_rep: torch.Tensor,
    y_rep: torch.Tensor,
    z_rep: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    outs = []
    total = t_rep.shape[0]
    for s in range(0, total, batch):
        e = min(s + batch, total)
        outs.append(pde.exact_solution(t_rep[s:e], x_rep[s:e], y_rep[s:e], z_rep[s:e]))
    return torch.cat(outs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate mean/var fields on a 2D grid."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["s2pinn", "vanilla"], required=True
    )
    parser.add_argument(
        "--pde",
        type=str,
        default=None,
        choices=["diffusion", "allen_cahn", "burgers", "darcy"],
    )
    parser.add_argument("--grid_res", type=int, default=64)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=262144)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)

    ckpt = _load_checkpoint(args.checkpoint, device)
    cfg: Optional[Dict[str, Any]] = None
    if args.config is not None:
        cfg = load_config(args.config)
    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
    else:
        raise ValueError("Need --config or a checkpoint containing a 'config' dict")

    if args.pde is not None:
        cfg.setdefault("pde", {})["type"] = args.pde

    if args.R is not None:
        cfg.setdefault("model", {}).setdefault("stochastic", {})["R"] = args.R
        cfg.setdefault("pde", {}).setdefault("kl", {})["n_terms"] = args.R

    pde = build_pde(cfg)
    r_dim = int(cfg["model"]["stochastic"]["R"])

    if args.model_type == "s2pinn":
        model = build_model(cfg, device)
    else:
        model = build_vanilla_model(
            R=r_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            activation=args.activation,
            dropout=float(args.dropout),
            device=device,
        )

    state = ckpt.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint missing model_state_dict")
    model.load_state_dict(state)
    model.eval()

    x_grid, y_grid = _grid_xy(cfg, args.grid_res, device, dtype)
    x_flat = x_grid.reshape(-1, 1)
    y_flat = y_grid.reshape(-1, 1)
    b = x_flat.shape[0]
    t_flat = torch.full((b, 1), float(args.t), device=device, dtype=dtype)

    z = pde.sample_z(int(args.N), device=device, dtype=dtype)
    t_rep, x_rep, y_rep, z_rep, b, nz = _expand_inputs(t_flat, x_flat, y_flat, z)

    u_pred = _forward_batched(
        model, t_rep, x_rep, y_rep, z_rep, int(args.batch)
    ).reshape(b, nz)
    u_ref = _exact_batched(pde, t_rep, x_rep, y_rep, z_rep, int(args.batch)).reshape(
        b, nz
    )

    mean_pred = u_pred.mean(dim=1).reshape(args.grid_res, args.grid_res)
    var_pred = u_pred.var(dim=1, unbiased=False).reshape(args.grid_res, args.grid_res)
    mean_ref = u_ref.mean(dim=1).reshape(args.grid_res, args.grid_res)
    var_ref = u_ref.var(dim=1, unbiased=False).reshape(args.grid_res, args.grid_res)

    out_dict = {
        "pde": cfg["pde"].get("type"),
        "model_type": args.model_type,
        "N": int(args.N),
        "grid_res": int(args.grid_res),
        "t": float(args.t),
        "mean_pred": mean_pred.cpu(),
        "var_pred": var_pred.cpu(),
        "mean_ref": mean_ref.cpu(),
        "var_ref": var_ref.cpu(),
        "x": x_grid.cpu(),
        "y": y_grid.cpu(),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out_dict, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
