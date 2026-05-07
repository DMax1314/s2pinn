"""Vanilla PINN baseline for all stochastic PDE problems.

Uses a standard MLP that takes (t, x, y, Z1..ZR) as input.
Same PDE, same evaluation, same WandB logging — only the model differs.
Supports all PDE types via build_pde().
"""

import argparse
import importlib
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))

_spinn_bases = importlib.import_module("spinn.models.bases")
_spinn_vanilla = importlib.import_module("spinn.models.vanilla_pinn")
_spinn_train = importlib.import_module("spinn.train")

GPCBasis = _spinn_bases.GPCBasis
VanillaPINN = _spinn_vanilla.VanillaPINN
_rel_l2 = _spinn_train._rel_l2
build_pde = _spinn_train.build_pde
compute_gpc_projection_loss = _spinn_train.compute_gpc_projection_loss
evaluate_uq = _spinn_train.evaluate_uq
load_config = _spinn_train.load_config
sample_batch = _spinn_train.sample_batch
set_seed = _spinn_train.set_seed


def build_vanilla_model(
    R: int,
    hidden_dim: int = 128,
    n_layers: int = 4,
    activation: str = "tanh",
    dropout: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> VanillaPINN:
    input_dim = 3 + R
    model = VanillaPINN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        dropout=dropout,
    )
    return model.to(device)


def vanilla_evaluate(
    model: VanillaPINN,
    pde: Any,
    eval_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Pointwise evaluation at fixed Z values + UQ moment evaluation."""
    model.eval()
    metrics: Dict[str, float] = {}
    grid_res = eval_cfg["grid_res"]

    with torch.no_grad():
        for z_info in eval_cfg["z_eval"]:
            z_name = z_info["name"]
            z_val = torch.tensor(z_info["values"], dtype=dtype, device=device)

            for t_val in eval_cfg["t_eval"]:
                grid = pde.eval_grid(grid_res, t_val, z_val, device=device, dtype=dtype)
                t_g = grid["t"].squeeze(-1)
                x_g = grid["x"].squeeze(-1)
                y_g = grid["y"].squeeze(-1)
                Z_g = grid["Z"]
                u_exact = grid["u_exact"].squeeze(-1)

                u_pred = model(t_g, x_g, y_g, Z_g)

                l2_err = torch.sqrt(((u_pred - u_exact) ** 2).sum())
                l2_ref = torch.sqrt((u_exact**2).sum()).clamp_min(1e-12)
                rel_l2 = (l2_err / l2_ref).item()

                metrics[f"eval_{z_name}_t{t_val}"] = rel_l2

    # UQ evaluation (moment-based)
    uq_cfg = eval_cfg.get("uq")
    if isinstance(uq_cfg, dict) and bool(uq_cfg.get("enabled", False)):
        metrics.update(evaluate_uq(model, pde, uq_cfg, device, dtype))

    model.train()
    return metrics


def mc_dropout_evaluate(
    model: VanillaPINN,
    pde: Any,
    eval_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    mc_samples: int = 50,
) -> Dict[str, float]:
    """MC-dropout uncertainty quantification."""
    was_training = model.training
    model.train()

    uq_cfg = eval_cfg.get("uq", {})
    n_probe = int(uq_cfg.get("n_probe", 2048))
    n_z_test = int(uq_cfg.get("n_z_test", 512))
    alpha = float(uq_cfg.get("alpha", 0.1))
    batch_limit = int(uq_cfg.get("batch_limit", 262144))

    probes = pde.sample_collocation(n_probe, device=device, dtype=dtype)
    t = probes["t"].squeeze(-1)
    x = probes["x"].squeeze(-1)
    y = probes["y"].squeeze(-1)

    Z_test = pde.sample_z(n_z_test, device=device, dtype=dtype)

    B = t.shape[0]
    nz = Z_test.shape[0]

    t_rep = t.unsqueeze(1).expand(B, nz).reshape(-1)
    x_rep = x.unsqueeze(1).expand(B, nz).reshape(-1)
    y_rep = y.unsqueeze(1).expand(B, nz).reshape(-1)
    Z_rep = Z_test.unsqueeze(0).expand(B, nz, -1).reshape(-1, Z_test.shape[1])

    all_preds = []
    total = t_rep.shape[0]
    with torch.no_grad():
        for _ in range(mc_samples):
            u_pred_all = []
            for s in range(0, total, batch_limit):
                e = min(s + batch_limit, total)
                u_pred_all.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e]))
            all_preds.append(torch.cat(u_pred_all, dim=0).reshape(B, nz))

    preds = torch.stack(all_preds, dim=0)

    mc_mean = preds.mean(dim=0)
    mc_var = preds.var(dim=0, unbiased=False)

    u_ref_list = []
    with torch.no_grad():
        for s in range(0, total, batch_limit):
            e = min(s + batch_limit, total)
            u_ref_list.append(
                pde.exact_solution(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e])
            )
    u_ref = torch.cat(u_ref_list, dim=0).reshape(B, nz)

    mean_pred_avg = mc_mean.mean(dim=1)
    mean_ref_avg = u_ref.mean(dim=1)
    var_pred_avg = mc_var.mean(dim=1)
    var_ref_avg = u_ref.var(dim=1, unbiased=False)

    metrics: Dict[str, float] = {}
    metrics["mc_uq/mean_pred"] = mc_mean.mean().item()
    metrics["mc_uq/var_pred"] = mc_var.mean().item()
    metrics["mc_uq/mean_rel_l2"] = _rel_l2(mean_pred_avg, mean_ref_avg).item()
    metrics["mc_uq/var_rel_l2"] = _rel_l2(var_pred_avg, var_ref_avg).item()

    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lo = torch.quantile(preds, lo_q, dim=0)
    hi = torch.quantile(preds, hi_q, dim=0)
    inside = (u_ref >= lo) & (u_ref <= hi)
    metrics["mc_uq/coverage"] = inside.float().mean().item()
    metrics["mc_uq/sharpness"] = (hi - lo).mean().item()

    if not was_training:
        model.eval()
    return metrics


def train_baseline(cfg: Dict[str, Any]) -> None:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = cfg["model"]["stochastic"]["R"]
    hidden_dim = cfg.get("baseline", {}).get("hidden_dim", 128)
    n_layers = cfg.get("baseline", {}).get("n_layers", 4)
    activation = cfg.get("baseline", {}).get("activation", "tanh")
    dropout = float(cfg.get("baseline", {}).get("dropout", 0.0))
    mc_samples = int(cfg.get("baseline", {}).get("mc_samples", 0))

    model = build_vanilla_model(R, hidden_dim, n_layers, activation, dropout, device)
    param_count = sum(p.numel() for p in model.parameters())

    # Use generic PDE builder (supports all PDE types)
    pde = build_pde(cfg)
    pde_type = cfg["pde"].get("type", "diffusion")

    # Auto-generate z_eval entries matching R dimension (same logic as train.py)
    z_evals = cfg["eval"].get("z_eval", [])
    z_dim_mismatch = any(len(z["values"]) != R for z in z_evals)
    if z_dim_mismatch or not z_evals:
        z_evals_new = [{"name": "Z0", "values": [0.0] * R}]
        for i in range(min(R, 2)):
            vals = [0.0] * R
            vals[i] = 1.0
            z_evals_new.append({"name": f"Z_e{i + 1}", "values": vals})
        cfg["eval"]["z_eval"] = z_evals_new

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    gpc_basis = GPCBasis(
        R=cfg["model"]["stochastic"]["R"],
        p=cfg["model"]["stochastic"]["p"],
        dist=cfg["model"]["stochastic"]["dist"],
    ).to(device)

    use_wandb = cfg["logging"].get("use_wandb", True)
    wandb_run = None
    if use_wandb:
        try:
            wandb = importlib.import_module("wandb")

            if dropout > 0.0:
                run_name = (
                    f"mcdrop_{pde_type}_h{hidden_dim}_L{n_layers}_R{R}"
                    f"_drop{dropout}_mc{mc_samples}_seed{seed}"
                )
            else:
                run_name = (
                    f"baseline_{pde_type}_h{hidden_dim}_L{n_layers}_R{R}_seed{seed}"
                )
            wandb_run = wandb.init(
                project=cfg["logging"].get("project", "s2pinn"),
                entity=cfg["logging"].get("entity"),
                name=run_name,
                config={
                    **cfg,
                    "model_type": "vanilla_pinn",
                    "param_count": param_count,
                },
                reinit=True,
            )
        except ImportError:
            print("WARNING: wandb not installed")
            use_wandb = False

    steps = train_cfg["steps"]
    n_colloc = train_cfg["colloc"]
    n_bc = train_cfg["bc"]
    n_ic = train_cfg["ic"]
    w_pde = train_cfg["w_pde"]
    w_proj = train_cfg["w_proj"]
    w_bc = train_cfg["w_bc"]
    w_ic = train_cfg["w_ic"]
    log_every = cfg["logging"].get("log_every", 10)
    eval_every = cfg["eval"].get("eval_every", 500)
    resample_every = train_cfg.get("resample_every", 1)

    batches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_metrics: Dict[str, float] = {}

    print(
        f"Training Vanilla PINN Baseline | {pde_type} | {param_count} params | {device}"
    )
    print(f"  hidden={hidden_dim}, layers={n_layers}, R={R}, w_proj={w_proj}")

    for step in range(steps):
        step_start = time.time()

        if batches is None or (resample_every > 0 and step % resample_every == 0):
            batches = sample_batch(pde, n_colloc, n_bc, n_ic, device, torch.float32)

        assert batches is not None

        optimizer.zero_grad(set_to_none=True)

        col = batches["colloc"]
        t_c = col["t"].squeeze(-1)
        x_c = col["x"].squeeze(-1)
        y_c = col["y"].squeeze(-1)
        Z_c = col["Z"]

        # Use generic forward_with_derivatives + pde.compute_residual
        derivs = model.forward_with_derivatives(t_c, x_c, y_c, Z_c)
        residual = pde.compute_residual(t_c, x_c, y_c, Z_c, derivs)

        loss_pde = (residual**2).mean()

        if w_proj != 0.0:
            loss_proj = compute_gpc_projection_loss(residual, Z_c, gpc_basis)
        else:
            loss_proj = residual.new_tensor(0.0)

        bc_b = batches["bc"]
        u_bc = model(
            bc_b["t"].squeeze(-1),
            bc_b["x"].squeeze(-1),
            bc_b["y"].squeeze(-1),
            bc_b["Z"],
        )
        u_bc_exact = pde.exact_solution(
            bc_b["t"].squeeze(-1),
            bc_b["x"].squeeze(-1),
            bc_b["y"].squeeze(-1),
            bc_b["Z"],
        )
        loss_bc = ((u_bc - u_bc_exact) ** 2).mean()

        ic_b = batches["ic"]
        u_ic = model(
            ic_b["t"].squeeze(-1),
            ic_b["x"].squeeze(-1),
            ic_b["y"].squeeze(-1),
            ic_b["Z"],
        )
        u_ic_exact = pde.exact_solution(
            ic_b["t"].squeeze(-1),
            ic_b["x"].squeeze(-1),
            ic_b["y"].squeeze(-1),
            ic_b["Z"],
        )
        loss_ic = ((u_ic - u_ic_exact) ** 2).mean()

        loss_total = (
            w_pde * loss_pde + w_proj * loss_proj + w_bc * loss_bc + w_ic * loss_ic
        )
        loss_total.backward()
        optimizer.step()

        step_time_ms = (time.time() - step_start) * 1000.0

        if step % log_every == 0:
            log_dict = {
                "step": step,
                "loss/total": loss_total.item(),
                "loss/pde": loss_pde.item(),
                "loss/proj": loss_proj.item(),
                "loss/bc": loss_bc.item(),
                "loss/ic": loss_ic.item(),
                "perf/step_time_ms": step_time_ms,
            }
            if use_wandb and wandb_run is not None:
                wandb_run.log(log_dict, step=step)

            if step % (log_every * 10) == 0:
                print(
                    f"[{step:>5d}/{steps}] "
                    f"total={loss_total.item():.4e} "
                    f"pde={loss_pde.item():.4e} "
                    f"proj={loss_proj.item():.4e} "
                    f"bc={loss_bc.item():.4e} "
                    f"ic={loss_ic.item():.4e} "
                    f"({step_time_ms:.0f}ms)"
                )

        if eval_every > 0 and (step + 1) % eval_every == 0:
            eval_metrics = vanilla_evaluate(
                model, pde, cfg["eval"], device, torch.float32
            )
            if mc_samples > 0 and getattr(model, "dropout_p", 0.0) > 0.0:
                eval_metrics.update(
                    mc_dropout_evaluate(
                        model,
                        pde,
                        cfg["eval"],
                        device,
                        torch.float32,
                        mc_samples=mc_samples,
                    )
                )
            for k_name, v in eval_metrics.items():
                if k_name not in best_metrics or v < best_metrics[k_name]:
                    best_metrics[k_name] = v

            if use_wandb and wandb_run is not None:
                wandb_run.log(eval_metrics, step=step)
                wandb_run.log(
                    {f"best/{k_name}": v for k_name, v in best_metrics.items()},
                    step=step,
                )

            print(
                f"  [EVAL step={step + 1}] "
                + "  ".join(f"{k_name}={v:.4e}" for k_name, v in eval_metrics.items())
            )

    final_metrics = vanilla_evaluate(model, pde, cfg["eval"], device, torch.float32)
    if mc_samples > 0 and getattr(model, "dropout_p", 0.0) > 0.0:
        final_metrics.update(
            mc_dropout_evaluate(
                model,
                pde,
                cfg["eval"],
                device,
                torch.float32,
                mc_samples=mc_samples,
            )
        )
    print(f"\n=== Final Evaluation (Vanilla PINN — {pde_type}) ===")
    for k_name, v in final_metrics.items():
        print(f"  {k_name}: {v:.6e}")

    if use_wandb and wandb_run is not None:
        wandb_run.log(
            {f"final/{k_name}": v for k_name, v in final_metrics.items()}, step=steps
        )
        wandb_run.finish()

    ckpt_dir = os.path.join(cfg["logging"].get("log_dir", "checkpoints"), "ckpt")
    if cfg["eval"].get("save_checkpoint", True):
        os.makedirs(ckpt_dir, exist_ok=True)
        pde_type = cfg["pde"].get("type", "diffusion")
        ckpt_path = os.path.join(
            ckpt_dir,
            f"vanilla_{pde_type}_h{hidden_dim}_L{n_layers}_drop{dropout}_seed{seed}.pt",
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "final_metrics": final_metrics,
                "best_metrics": best_metrics,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla PINN baseline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument(
        "--pde",
        type=str,
        default=None,
        choices=["diffusion", "allen_cahn", "burgers", "darcy", "poisson_mc", "darcy_mc"],
    )
    parser.add_argument("--w_proj", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--uq", action="store_true", help="Enable UQ evaluation")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability for MC-dropout (0=disabled)",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=0,
        help="MC forward passes for dropout uncertainty (0=disabled)",
    )
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)

    cfg.setdefault("baseline", {})
    cfg["baseline"]["hidden_dim"] = args.hidden_dim
    cfg["baseline"]["n_layers"] = args.n_layers
    cfg["baseline"]["activation"] = args.activation
    cfg["baseline"]["dropout"] = args.dropout
    cfg["baseline"]["mc_samples"] = args.mc_samples

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg["training"]["steps"] = args.steps
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.R is not None:
        cfg["model"]["stochastic"]["R"] = args.R
        cfg["pde"]["kl"]["n_terms"] = args.R
    if args.pde is not None:
        cfg["pde"]["type"] = args.pde
    if args.w_proj is not None:
        cfg["training"]["w_proj"] = args.w_proj
    if args.gamma is not None:
        cfg.setdefault("pde", {}).setdefault("darcy", {})["gamma"] = args.gamma
    if args.uq:
        cfg.setdefault("eval", {}).setdefault("uq", {})["enabled"] = True
    if args.no_wandb:
        cfg["logging"]["use_wandb"] = False

    train_baseline(cfg)


if __name__ == "__main__":
    main()
