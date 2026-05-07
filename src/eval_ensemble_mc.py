"""Deep Ensemble evaluation on non-manufactured MC benchmarks.

Loads K vanilla PINN checkpoints, pools predictions across members and Z-samples,
then compares mean/var against FDM Monte Carlo reference.
"""

import argparse
import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.dirname(__file__))

from spinn.models.vanilla_pinn import VanillaPINN
from spinn.train import build_pde, load_config, set_seed


def _build_vanilla(R: int, h: int, L: int, device: torch.device) -> VanillaPINN:
    return VanillaPINN(
        input_dim=3 + R,
        hidden_dim=h,
        n_layers=L,
        activation="tanh",
        dropout=0.0,
    ).to(device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Ensemble eval on MC benchmarks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--pde_type", type=str, required=True, choices=["poisson_mc", "darcy_mc"]
    )
    parser.add_argument("--mean_ref", type=str, required=True)
    parser.add_argument("--var_ref", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/ckpt")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024]
    )
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--N", type=int, default=10000, help="Number of Z samples")
    parser.add_argument("--batch", type=int, default=262144)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    cfg = load_config(args.config)
    cfg["pde"]["type"] = args.pde_type
    cfg["model"]["stochastic"]["R"] = args.R
    cfg["pde"]["kl"]["n_terms"] = args.R

    pde = build_pde(cfg)

    # Build checkpoint paths
    seeds = args.seeds[: args.n_members]
    ckpt_paths: List[str] = []
    for seed in seeds:
        fname = f"vanilla_{args.pde_type}_h{args.hidden_dim}_L{args.n_layers}_drop0.0_seed{seed}.pt"
        ckpt_paths.append(os.path.join(args.ckpt_dir, fname))

    # Load ensemble members
    models: List[VanillaPINN] = []
    for path in ckpt_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device)
        model = _build_vanilla(args.R, args.hidden_dim, args.n_layers, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"Loaded member: {os.path.basename(path)}")

    print(f"\nEnsemble: K={len(models)} members")

    # Load FDM MC reference
    mean_ref = torch.load(args.mean_ref, map_location=device)
    var_ref = torch.load(args.var_ref, map_location=device)
    grid_res = mean_ref.shape[0]
    print(f"Reference grid: {grid_res}x{grid_res}")

    # Build spatial grid
    x_lin = torch.linspace(-1.0, 1.0, grid_res, device=device, dtype=dtype)
    y_lin = torch.linspace(-1.0, 1.0, grid_res, device=device, dtype=dtype)
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x_flat = x_grid.reshape(-1, 1)
    y_flat = y_grid.reshape(-1, 1)

    B = x_flat.shape[0]
    t_flat = torch.full((B, 1), 1.0, device=device, dtype=dtype)

    # Sample Z
    Z = pde.sample_z(args.N, device=device, dtype=dtype)
    nz = Z.shape[0]

    # Expand for all (grid_point, Z) pairs
    t_rep = t_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    x_rep = x_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    y_rep = y_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    Z_rep = Z.unsqueeze(0).expand(B, nz, -1).reshape(-1, Z.shape[1])

    total = t_rep.shape[0]
    print(f"Evaluating {B} grid points x {nz} Z-samples x {len(models)} members...")

    # Collect predictions from all ensemble members: shape [K, B, nz]
    all_member_preds: List[torch.Tensor] = []
    for k, model in enumerate(models):
        u_pred_all = []
        for s in range(0, total, args.batch):
            e = min(s + args.batch, total)
            u_pred_all.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e]))
        member_pred = torch.cat(u_pred_all, dim=0).reshape(B, nz)
        all_member_preds.append(member_pred)
        print(f"  Member {k}: done")

    # Pool across members: shape [B, K*nz]
    stacked = torch.stack(all_member_preds, dim=1)  # [B, K, nz]
    pooled = stacked.reshape(B, len(models) * nz)  # [B, K*nz]

    mean_pred = pooled.mean(dim=1).reshape(grid_res, grid_res)
    var_pred = pooled.var(dim=1, unbiased=False).reshape(grid_res, grid_res)

    def rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
        err = torch.sqrt(((pred - ref) ** 2).sum())
        norm = torch.sqrt((ref**2).sum()).clamp_min(1e-12)
        return (err / norm).item()

    mean_err = rel_l2(mean_pred, mean_ref)
    var_err = rel_l2(var_pred, var_ref)

    print("\n=== RESULTS ===")
    print(f"PDE type:               {args.pde_type}")
    print(f"Ensemble members:       {len(models)}")
    print(f"Mean Relative L2 Error: {mean_err:.4e}")
    print(f"Var Relative L2 Error:  {var_err:.4e}")

    # Save predictions for potential heatmap plotting
    res_dict = {
        "mean_pred": mean_pred.cpu(),
        "var_pred": var_pred.cpu(),
        "mean_ref": mean_ref.cpu(),
        "var_ref": var_ref.cpu(),
        "x": x_grid.cpu(),
        "y": y_grid.cpu(),
    }
    out_name = f"{args.pde_type}_ensemble_predictions.pt"
    torch.save(res_dict, out_name)
    print(f"Saved fields to {out_name}")


if __name__ == "__main__":
    main()
