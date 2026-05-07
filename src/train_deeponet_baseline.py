import argparse
import time
from typing import Any, Dict, Optional

import torch

from spinn.models.bases import GPCBasis
from spinn.models.deeponet import PIDeepONet
from spinn.train import build_pde, evaluate, load_config, sample_batch, set_seed


def train(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    R = int(cfg["model"]["stochastic"]["R"])
    de_cfg = cfg.get("deeponet", {})
    model = PIDeepONet(
        R=R,
        latent_dim=int(de_cfg.get("latent_dim", 128)),
        branch_width=int(de_cfg.get("branch_width", 128)),
        branch_depth=int(de_cfg.get("branch_depth", 3)),
        trunk_width=int(de_cfg.get("trunk_width", 128)),
        trunk_depth=int(de_cfg.get("trunk_depth", 3)),
        activation=str(de_cfg.get("activation", "tanh")),
    ).to(device)

    pde = build_pde(cfg)
    pde_type = cfg.get("pde", {}).get("type", "diffusion")

    # Auto-generate z_eval entries matching R dimension (same logic as train.py)
    z_evals = cfg.get("eval", {}).get("z_eval", [])
    z_dim_mismatch = any(len(z["values"]) != R for z in z_evals)
    if z_dim_mismatch or not z_evals:
        z_evals_new = [{"name": "Z0", "values": [0.0] * R}]
        for i in range(min(R, 2)):
            vals = [0.0] * R
            vals[i] = 1.0
            z_evals_new.append({"name": f"Z_e{i + 1}", "values": vals})
        cfg.setdefault("eval", {})["z_eval"] = z_evals_new

    train_cfg = cfg["training"]
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    gpc_basis = GPCBasis(
        R=R,
        p=int(cfg["model"]["stochastic"]["p"]),
        dist=str(cfg["model"]["stochastic"]["dist"]),
    ).to(device)

    use_wandb = bool(cfg.get("logging", {}).get("use_wandb", True))
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = f"deeponet_{pde_type}_R{R}_seed{seed}"
            wandb_run = wandb.init(
                project=cfg.get("logging", {}).get("project", "s2pinn"),
                entity=cfg.get("logging", {}).get("entity"),
                name=run_name,
                config=cfg,
                reinit=True,
            )
        except Exception:
            use_wandb = False

    steps = int(train_cfg["steps"])
    n_colloc = int(train_cfg["colloc"])
    n_bc = int(train_cfg["bc"])
    n_ic = int(train_cfg["ic"])
    resample_every = int(train_cfg.get("resample_every", 1))

    w_pde = float(train_cfg["w_pde"])
    w_proj = float(train_cfg["w_proj"])
    w_bc = float(train_cfg["w_bc"])
    w_ic = float(train_cfg["w_ic"])

    log_every = int(cfg.get("logging", {}).get("log_every", 10))
    eval_every = int(cfg.get("eval", {}).get("eval_every", 500))

    batches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_metrics: Dict[str, float] = {}

    for step in range(steps):
        start = time.time()
        if batches is None or (resample_every > 0 and step % resample_every == 0):
            batches = sample_batch(pde, n_colloc, n_bc, n_ic, device, dtype)

        opt.zero_grad(set_to_none=True)

        col = batches["colloc"]
        t = col["t"].squeeze(-1)
        x = col["x"].squeeze(-1)
        y = col["y"].squeeze(-1)
        Z = col["Z"]

        derivs = model.forward_with_derivatives(t, x, y, Z)
        residual = pde.compute_residual(t, x, y, Z, derivs)
        loss_pde = (residual**2).mean()
        if w_proj != 0.0:
            Psi = gpc_basis(Z)
            proj_means = (residual.unsqueeze(1) * Psi).mean(dim=0)
            loss_proj = (proj_means**2).sum()
        else:
            loss_proj = residual.new_tensor(0.0)

        bc = batches["bc"]
        u_bc = model(
            bc["t"].squeeze(-1),
            bc["x"].squeeze(-1),
            bc["y"].squeeze(-1),
            bc["Z"],
        )
        u_bc_ref = pde.exact_solution(
            bc["t"].squeeze(-1),
            bc["x"].squeeze(-1),
            bc["y"].squeeze(-1),
            bc["Z"],
        )
        loss_bc = ((u_bc - u_bc_ref) ** 2).mean()

        ic = batches["ic"]
        u_ic = model(
            ic["t"].squeeze(-1),
            ic["x"].squeeze(-1),
            ic["y"].squeeze(-1),
            ic["Z"],
        )
        u_ic_ref = pde.exact_solution(
            ic["t"].squeeze(-1),
            ic["x"].squeeze(-1),
            ic["y"].squeeze(-1),
            ic["Z"],
        )
        loss_ic = ((u_ic - u_ic_ref) ** 2).mean()

        loss = w_pde * loss_pde + w_proj * loss_proj + w_bc * loss_bc + w_ic * loss_ic
        loss.backward()
        opt.step()

        step_ms = (time.time() - start) * 1000.0
        if log_every > 0 and step % log_every == 0:
            log_dict = {
                "step": step,
                "loss/total": loss.item(),
                "loss/pde": loss_pde.item(),
                "loss/proj": loss_proj.item(),
                "loss/bc": loss_bc.item(),
                "loss/ic": loss_ic.item(),
                "perf/step_time_ms": step_ms,
            }
            if use_wandb and wandb_run is not None:
                import wandb  # type: ignore[import-untyped]

                wandb.log(log_dict, step=step)
            if step % (log_every * 10) == 0:
                print(
                    f"[{step:>5d}/{steps}] total={loss.item():.4e} pde={loss_pde.item():.4e} "
                    f"proj={loss_proj.item():.4e} bc={loss_bc.item():.4e} ic={loss_ic.item():.4e} ({step_ms:.0f}ms)"
                )

        if eval_every > 0 and (step + 1) % eval_every == 0:
            eval_metrics = evaluate(model, pde, cfg["eval"], device, dtype)
            for k, v in eval_metrics.items():
                if k not in best_metrics or v < best_metrics[k]:
                    best_metrics[k] = v
            if use_wandb and wandb_run is not None:
                import wandb  # type: ignore[import-untyped]

                wandb.log(eval_metrics, step=step)
                wandb.log({f"best/{k}": v for k, v in best_metrics.items()}, step=step)
            print(
                f"  [EVAL step={step + 1}] "
                + "  ".join(f"{k}={v:.4e}" for k, v in eval_metrics.items())
            )

    final_metrics = evaluate(model, pde, cfg["eval"], device, dtype)
    print("\n=== Final Evaluation ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6e}")

    if use_wandb and wandb_run is not None:
        import wandb  # type: ignore[import-untyped]

        wandb.log({f"final/{k}": v for k, v in final_metrics.items()}, step=steps)
        wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--w_proj", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--uq", action="store_true", help="Enable UQ evaluation")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.R is not None:
        cfg["model"]["stochastic"]["R"] = int(args.R)
        cfg["pde"]["kl"]["n_terms"] = int(args.R)
    if args.w_proj is not None:
        cfg["training"]["w_proj"] = float(args.w_proj)
    if args.gamma is not None:
        cfg.setdefault("pde", {}).setdefault("darcy", {})["gamma"] = float(args.gamma)
    if args.uq:
        cfg.setdefault("eval", {}).setdefault("uq", {})["enabled"] = True
    if args.no_wandb:
        cfg.setdefault("logging", {})["use_wandb"] = False
    train(cfg)


if __name__ == "__main__":
    main()
