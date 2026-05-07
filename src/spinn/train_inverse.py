import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

from spinn.models.bases import GPCBasis
from spinn.models.product_model import ProductModelGPC
from spinn.pdes.inverse_stochastic_diffusion_2d import InverseStochasticDiffusion2D
from spinn.pdes.inverse_stochastic_burgers_2d import InverseStochasticBurgers2D

PDEType = Union[
    InverseStochasticDiffusion2D,
    InverseStochasticBurgers2D,
]

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any], device: torch.device) -> ProductModelGPC:
    spatial_cfg = {
        "n_atoms": cfg["model"]["spatial"]["n_atoms"],
        "init_sigma": cfg["model"]["spatial"]["init_sigma"],
        "use_warp": cfg["model"]["spatial"]["use_warp"],
        "warp_hidden": cfg["model"]["spatial"]["warp_hidden"],
        "warp_eps": cfg["model"]["spatial"]["warp_eps"],
    }
    temporal_cfg = {"n_freq": cfg["model"]["temporal"]["n_freq"]}
    stochastic_cfg = {
        "R": cfg["model"]["stochastic"]["R"],
        "p": cfg["model"]["stochastic"]["p"],
        "dist": cfg["model"]["stochastic"]["dist"],
    }
    cp_cfg = {"rank": cfg["model"]["cp"]["rank"]}

    model = ProductModelGPC(
        spatial_cfg=spatial_cfg,
        temporal_cfg=temporal_cfg,
        stochastic_cfg=stochastic_cfg,
        cp_cfg=cp_cfg,
    )
    return model.to(device)


def build_pde(
    cfg: Dict[str, Any],
) -> PDEType:
    pde_cfg = cfg["pde"]
    pde_type = pde_cfg.get("type", "diffusion")

    common_args = {
        "R": cfg["model"]["stochastic"]["R"],
        "sigma": pde_cfg["kl"]["sigma"],
        "corr_length": pde_cfg["kl"]["corr_length"],
        "x_range": (pde_cfg["domain"]["x_min"], pde_cfg["domain"]["x_max"]),
        "y_range": (pde_cfg["domain"]["y_min"], pde_cfg["domain"]["y_max"]),
        "t_range": (pde_cfg["domain"]["t_min"], pde_cfg["domain"]["t_max"]),
        "z_distribution": cfg["model"]["stochastic"]["dist"],
    }

    if pde_type == "inverse_diffusion":
        # Initial guesses for learnable parameters (true: mu_k=0.0, sigma=1.0)
        init_mu_k = pde_cfg.get("init_mu_k", 0.5)
        init_sigma = pde_cfg.get("init_sigma", 0.5)
        # Override KL sigma with learnable initial guess (KL modes don't depend on sigma)
        common_args["sigma"] = init_sigma
        return InverseStochasticDiffusion2D(**common_args, mu_k=init_mu_k)
    elif pde_type == "inverse_burgers":
        init_mu_k = pde_cfg.get("init_mu_k", 0.5)
        init_sigma = pde_cfg.get("init_sigma", 0.5)
        common_args["sigma"] = init_sigma
        return InverseStochasticBurgers2D(**common_args, mu_k=init_mu_k)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def sample_batch(
    pde: PDEType,
    n_colloc: int,
    n_bc: int,
    n_ic: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Dict[str, torch.Tensor]]:
    colloc = pde.sample_collocation(n_colloc, device=device, dtype=dtype)
    bc = pde.sample_boundary(n_bc, device=device, dtype=dtype)
    ic = pde.sample_initial(n_ic, device=device, dtype=dtype)
    return {"colloc": colloc, "bc": bc, "ic": ic}


def compute_pde_residual(
    model: ProductModelGPC,
    pde: PDEType,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = batch["t"].squeeze(-1)
    x = batch["x"].squeeze(-1).requires_grad_(True)
    y = batch["y"].squeeze(-1).requires_grad_(True)
    Z = batch["Z"]

    derivs = model.forward_with_derivatives(t, x, y, Z)
    u = derivs["u"]

    if hasattr(pde, "compute_residual"):
        residual = pde.compute_residual(t, x, y, Z, derivs)
    else:
        u_t = derivs["u_t"]
        laplacian_u = derivs["laplacian_u"]
        k = pde.diffusion_coefficient(x, y, Z)
        s = pde.source_term(t, x, y, Z)
        residual = u_t - k * laplacian_u - s

    return residual, u


def compute_gpc_projection_loss(
    residual: torch.Tensor,
    Z: torch.Tensor,
    gpc_basis: GPCBasis,
) -> torch.Tensor:
    Psi = gpc_basis(Z)
    r_times_psi = residual.unsqueeze(1) * Psi
    proj_means = r_times_psi.mean(dim=0)
    return (proj_means**2).sum()


def compute_bc_loss(
    model: ProductModelGPC,
    pde: PDEType,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    t = batch["t"].squeeze(-1)
    x = batch["x"].squeeze(-1)
    y = batch["y"].squeeze(-1)
    Z = batch["Z"]

    u_pred = model(t, x, y, Z)
    u_exact = pde.exact_solution(t, x, y, Z)
    return ((u_pred - u_exact) ** 2).mean()


def compute_data_loss(
    model: ProductModelGPC,
    obs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    t = obs["t"].squeeze(-1)
    x = obs["x"].squeeze(-1)
    y = obs["y"].squeeze(-1)
    Z = obs["Z"]
    u_true = obs["u"]

    u_pred = model(t, x, y, Z)
    return ((u_pred - u_true) ** 2).mean()

def compute_ic_loss(
    model: ProductModelGPC,
    pde: PDEType,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    t = batch["t"].squeeze(-1)
    x = batch["x"].squeeze(-1)
    y = batch["y"].squeeze(-1)
    Z = batch["Z"]

    u_pred = model(t, x, y, Z)
    u_exact = pde.exact_solution(t, x, y, Z)
    return ((u_pred - u_exact) ** 2).mean()


def _rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    err = torch.linalg.norm(pred - ref)
    denom = torch.linalg.norm(ref).clamp_min(1e-12)
    return err / denom


@torch.no_grad()
def evaluate_uq(
    model: Any,
    pde: Any,
    uq_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    # InverseStochasticDiffusion2D always has exact_solution, no skip needed

    model.eval()
    n_probe = int(uq_cfg.get("n_probe", 2048))
    n_z_moment = int(uq_cfg.get("n_z_moment", 1024))
    n_z_interval = int(uq_cfg.get("n_z_interval", 1024))
    n_z_test = int(uq_cfg.get("n_z_test", 1024))
    alpha = float(uq_cfg.get("alpha", 0.1))
    batch_limit = int(uq_cfg.get("batch_limit", 262144))

    probes = pde.sample_collocation(n_probe, device=device, dtype=dtype)
    t = probes["t"].squeeze(-1)
    x = probes["x"].squeeze(-1)
    y = probes["y"].squeeze(-1)
    B = t.shape[0]

    def moments_for(
        Z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nz = Z.shape[0]
        t_rep = t.unsqueeze(1).expand(B, nz).reshape(-1)
        x_rep = x.unsqueeze(1).expand(B, nz).reshape(-1)
        y_rep = y.unsqueeze(1).expand(B, nz).reshape(-1)
        Z_rep = Z.unsqueeze(0).expand(B, nz, -1).reshape(-1, Z.shape[1])

        u_pred_all = []
        u_ref_all = []
        total = t_rep.shape[0]
        for s in range(0, total, batch_limit):
            e = min(s + batch_limit, total)
            u_pred_all.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e]))
            u_ref_all.append(
                pde.exact_solution(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e])
            )
        u_pred = torch.cat(u_pred_all, dim=0).reshape(B, nz)
        u_ref = torch.cat(u_ref_all, dim=0).reshape(B, nz)

        mean_pred = u_pred.mean(dim=1)
        var_pred = u_pred.var(dim=1, unbiased=False)
        mean_ref = u_ref.mean(dim=1)
        var_ref = u_ref.var(dim=1, unbiased=False)
        return mean_pred, var_pred, mean_ref, var_ref

    Z_m = pde.sample_z(n_z_moment, device=device, dtype=dtype)
    mean_pred, var_pred, mean_ref, var_ref = moments_for(Z_m)

    metrics: Dict[str, float] = {}
    metrics["uq/mean_rel_l2"] = _rel_l2(mean_pred, mean_ref).item()
    metrics["uq/var_rel_l2"] = _rel_l2(var_pred, var_ref).item()

    if n_z_interval > 0 and n_z_test > 0:
        Z_int = pde.sample_z(n_z_interval, device=device, dtype=dtype)
        nz = Z_int.shape[0]
        t_rep = t.unsqueeze(1).expand(B, nz).reshape(-1)
        x_rep = x.unsqueeze(1).expand(B, nz).reshape(-1)
        y_rep = y.unsqueeze(1).expand(B, nz).reshape(-1)
        Z_rep = Z_int.unsqueeze(0).expand(B, nz, -1).reshape(-1, Z_int.shape[1])

        u_pred_all = []
        total = t_rep.shape[0]
        for s in range(0, total, batch_limit):
            e = min(s + batch_limit, total)
            u_pred_all.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e]))
        u_pred = torch.cat(u_pred_all, dim=0).reshape(B, nz)

        lo_q = float(alpha / 2.0)
        hi_q = float(1.0 - alpha / 2.0)
        lo = torch.quantile(u_pred, lo_q, dim=1)
        hi = torch.quantile(u_pred, hi_q, dim=1)

        Z_test = pde.sample_z(n_z_test, device=device, dtype=dtype)
        nz2 = Z_test.shape[0]
        t_rep2 = t.unsqueeze(1).expand(B, nz2).reshape(-1)
        x_rep2 = x.unsqueeze(1).expand(B, nz2).reshape(-1)
        y_rep2 = y.unsqueeze(1).expand(B, nz2).reshape(-1)
        Z_rep2 = Z_test.unsqueeze(0).expand(B, nz2, -1).reshape(-1, Z_test.shape[1])

        u_ref_all = []
        total2 = t_rep2.shape[0]
        for s in range(0, total2, batch_limit):
            e = min(s + batch_limit, total2)
            u_ref_all.append(
                pde.exact_solution(t_rep2[s:e], x_rep2[s:e], y_rep2[s:e], Z_rep2[s:e])
            )
        u_ref = torch.cat(u_ref_all, dim=0).reshape(B, nz2)

        inside = (u_ref >= lo.unsqueeze(1)) & (u_ref <= hi.unsqueeze(1))
        metrics["uq/coverage"] = inside.float().mean().item()

        for alpha_level in [0.5, 0.2, 0.1, 0.05]:
            nominal = 1.0 - alpha_level
            lo_q_ml = float(alpha_level / 2.0)
            hi_q_ml = float(1.0 - alpha_level / 2.0)
            lo_ml = torch.quantile(u_pred, lo_q_ml, dim=1)
            hi_ml = torch.quantile(u_pred, hi_q_ml, dim=1)
            inside_ml = (u_ref >= lo_ml.unsqueeze(1)) & (u_ref <= hi_ml.unsqueeze(1))
            cov = inside_ml.float().mean().item()
            width = (hi_ml - lo_ml).mean().item()
            pct = int(nominal * 100)
            metrics[f"uq/coverage_{pct}"] = cov
            metrics[f"uq/sharpness_{pct}"] = width

        cal_levels = [0.5, 0.2, 0.1, 0.05]
        cal_errors: List[float] = []
        for alpha_level in cal_levels:
            nominal = 1.0 - alpha_level
            pct = int(nominal * 100)
            empirical = metrics[f"uq/coverage_{pct}"]
            cal_errors.append(abs(nominal - empirical))
        metrics["uq/calibration_error"] = sum(cal_errors) / len(cal_errors)

    model.train()
    return metrics


def evaluate(
    model: ProductModelGPC,
    pde: PDEType,
    eval_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    grid_res = eval_cfg["grid_res"]
    t_evals = eval_cfg["t_eval"]
    z_evals = eval_cfg["z_eval"]

    with torch.no_grad():
        for z_info in z_evals:
            z_name = z_info["name"]
            z_val = torch.tensor(z_info["values"], dtype=dtype, device=device)

            for t_val in t_evals:
                grid = pde.eval_grid(grid_res, t_val, z_val, device=device, dtype=dtype)
                t_g = grid["t"].squeeze(-1)
                x_g = grid["x"].squeeze(-1)
                y_g = grid["y"].squeeze(-1)
                Z_g = grid["Z"]
                u_exact = grid["u_exact"].squeeze(-1)

                u_pred = model(t_g, x_g, y_g, Z_g)

                # Always compute rel_l2 (no MC-only PDE in inverse script)
                l2_err = torch.sqrt(((u_pred - u_exact) ** 2).sum())
                l2_ref = torch.sqrt((u_exact**2).sum()).clamp_min(1e-12)
                rel_l2 = (l2_err / l2_ref).item()

                key = f"eval_{z_name}_t{t_val}"
                metrics[key] = rel_l2

    uq_cfg = eval_cfg.get("uq")
    if isinstance(uq_cfg, dict) and bool(uq_cfg.get("enabled", False)):
        metrics.update(evaluate_uq(model, pde, uq_cfg, device, dtype))

    model.train()
    return metrics


def train(cfg: Dict[str, Any]) -> None:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode = cfg["training"].get("amp", "none")
    use_amp = amp_mode in ("bf16", "fp16") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    compute_dtype = torch.float32

    model = build_model(cfg, device)
    pde = build_pde(cfg)
    if isinstance(pde, torch.nn.Module):
        pde = pde.to(device)

    # Generate fixed observations for the inverse problem
    n_obs = cfg["training"].get("n_obs", 1000)
    obs_points = pde.sample_collocation(n_obs, device=device, dtype=compute_dtype)
    with torch.no_grad():
        obs_u = pde.exact_solution(obs_points["t"].squeeze(-1), obs_points["x"].squeeze(-1), obs_points["y"].squeeze(-1), obs_points["Z"])
        obs_points["u"] = obs_u

    # Auto-generate z_eval entries matching the current R dimension.
    # The config may hardcode z_eval for R=3 (default), but when R is
    # changed via CLI (--R), the eval vectors must be regenerated to
    # avoid a dimension mismatch crash in pde.eval_grid().
    R = cfg["model"]["stochastic"]["R"]
    z_evals = cfg["eval"].get("z_eval", [])
    z_dim_mismatch = any(len(z["values"]) != R for z in z_evals)
    if z_dim_mismatch or not z_evals:
        z_evals_new = [{"name": "Z0", "values": [0.0] * R}]
        for i in range(min(R, 2)):
            vals = [0.0] * R
            vals[i] = 1.0
            z_evals_new.append({"name": f"Z_e{i + 1}", "values": vals})
        cfg["eval"]["z_eval"] = z_evals_new

    param_count = sum(p.numel() for p in model.parameters())

    train_cfg = cfg["training"]
    base_lr = train_cfg["lr"]
    param_lr_mult = train_cfg.get("param_lr_mult", 10.0)  # PDE params get higher LR

    # Separate param groups: model weights at base LR, PDE params at higher LR
    param_groups = [{"params": list(model.parameters()), "lr": base_lr}]
    if isinstance(pde, torch.nn.Module) and len(list(pde.parameters())) > 0:
        param_groups.append({"params": list(pde.parameters()), "lr": base_lr * param_lr_mult})
        print(f"  PDE param LR: {base_lr * param_lr_mult:.1e} (model LR: {base_lr:.1e}, mult: {param_lr_mult}x)")
    optimizer = torch.optim.Adam(param_groups)

    # GradScaler needed for fp16 AMP (not for bf16 — sufficient dynamic range)
    scaler: Optional[torch.cuda.amp.GradScaler] = None  # type: ignore[attr-defined]
    if use_amp and amp_mode == "fp16":
        scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

    use_wandb = cfg["logging"].get("use_wandb", True)
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = (
                f"R{cfg['model']['stochastic']['R']}"
                f"_p{cfg['model']['stochastic']['p']}"
                f"_rank{cfg['model']['cp']['rank']}"
                f"_M{cfg['model']['spatial']['n_atoms']}"
                f"_seed{seed}"
            )
            wandb_run = wandb.init(
                project=cfg["logging"].get("project", "s2pinn"),
                entity=cfg["logging"].get("entity"),
                name=run_name,
                config=cfg,
                reinit=True,
            )
        except ImportError:
            print("WARNING: wandb not installed, logging to console only")
            use_wandb = False

    steps = train_cfg["steps"]
    n_colloc = train_cfg["colloc"]
    n_bc = train_cfg["bc"]
    n_ic = train_cfg["ic"]
    resample_every = train_cfg.get("resample_every", 1)
    normalize_cp_every = train_cfg.get("normalize_cp_every", 1)
    log_every = cfg["logging"].get("log_every", 10)
    eval_every = cfg["eval"].get("eval_every", 500)

    w_data = train_cfg.get("w_data", 10.0)
    w_pde = train_cfg["w_pde"]
    w_proj = train_cfg["w_proj"]
    w_bc = train_cfg["w_bc"]
    w_ic = train_cfg["w_ic"]
    w_orth = train_cfg["w_orth"]
    w_fac_orth = train_cfg["w_fac_orth"]

    gpc_basis = model.stochastic

    batches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    print(f"Training s2PINN | {param_count} params | {device} | AMP={amp_mode}")
    print(
        f"  R={cfg['model']['stochastic']['R']}, p={cfg['model']['stochastic']['p']}, "
        f"G={gpc_basis.num_basis}, rank={cfg['model']['cp']['rank']}, M={cfg['model']['spatial']['n_atoms']}"
    )

    best_metrics: Dict[str, float] = {}

    for step in range(steps):
        step_start = time.time()

        if batches is None or (resample_every > 0 and step % resample_every == 0):
            batches = sample_batch(pde, n_colloc, n_bc, n_ic, device, compute_dtype)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):  # type: ignore[attr-defined]
            residual, _ = compute_pde_residual(model, pde, batches["colloc"])

            loss_pde = (residual**2).mean()
            loss_proj = compute_gpc_projection_loss(
                residual, batches["colloc"]["Z"], gpc_basis
            )
            loss_bc = compute_bc_loss(model, pde, batches["bc"])
            loss_ic = compute_ic_loss(model, pde, batches["ic"])

            F_space_gram = model.spatial(
                batches["colloc"]["x"].squeeze(-1).detach(),
                batches["colloc"]["y"].squeeze(-1).detach(),
            )
            reg = model.regularization(F_space_gram, w_orth, w_fac_orth)

            loss_data = compute_data_loss(model, obs_points)

            loss_total = (
                w_pde * loss_pde
                + w_proj * loss_proj
                + w_bc * loss_bc
                + w_ic * loss_ic
                + w_data * loss_data
                + reg
            )

        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        if normalize_cp_every > 0 and (step + 1) % normalize_cp_every == 0:
            model.cp.normalize_factors()

        step_time_ms = (time.time() - step_start) * 1000.0

        if step % log_every == 0:
            log_dict = {
                "step": step,
                "loss/total": loss_total.item(),
                "loss/pde": loss_pde.item(),
                "loss/proj": loss_proj.item(),
                "loss/bc": loss_bc.item(),
                "loss/ic": loss_ic.item(),
                "loss/reg": reg.item(),
                "perf/step_time_ms": step_time_ms,
            }

            if hasattr(pde, "mu_k"):
                log_dict["param/mu_k"] = pde.mu_k.item()
                log_dict["param/sigma"] = pde.sigma.item()
                log_dict["error/mu_k_err"] = abs(pde.mu_k.item() - pde.true_mu_k)
                log_dict["error/sigma_err"] = abs(pde.sigma.item() - pde.true_sigma)
                log_dict["loss/data"] = loss_data.item()

            if torch.cuda.is_available():
                log_dict["perf/mem_MB"] = torch.cuda.max_memory_allocated() / (1024**2)

            if use_wandb and wandb_run is not None:
                import wandb  # type: ignore[import-untyped]

                wandb.log(log_dict, step=step)

            if step % (log_every * 10) == 0:
                param_str = ""
                if hasattr(pde, "mu_k"):
                    param_str = (
                        f" mu_k={pde.mu_k.item():.4f}"
                        f" sigma={pde.sigma.item():.4f}"
                    )
                print(
                    f"[{step:>5d}/{steps}] "
                    f"total={loss_total.item():.4e} "
                    f"pde={loss_pde.item():.4e} "
                    f"data={loss_data.item():.4e} "
                    f"reg={reg.item():.4e}"
                    f"{param_str} "
                    f"({step_time_ms:.0f}ms)"
                )

        if eval_every > 0 and (step + 1) % eval_every == 0:
            eval_metrics = evaluate(model, pde, cfg["eval"], device, compute_dtype)
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

    # ── L-BFGS phase (optional) ─────────────────────────────────────────────
    lbfgs_steps = train_cfg.get("lbfgs_steps", 0)
    if lbfgs_steps > 0:
        lbfgs_keep_reg = train_cfg.get("lbfgs_keep_reg", False)
        print(f"\n=== L-BFGS phase: {lbfgs_steps} steps | keep_reg={lbfgs_keep_reg} ===")
        all_params = list(model.parameters())
        if isinstance(pde, torch.nn.Module):
            all_params += list(pde.parameters())
        lbfgs_optimizer = torch.optim.LBFGS(
            all_params,
            max_iter=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        # Fix a single batch for L-BFGS (re-sampling breaks Hessian history)
        lbfgs_batches = sample_batch(pde, n_colloc, n_bc, n_ic, device, compute_dtype)

        # Cache last closure values for logging (avoids re-running inside torch.no_grad)
        _log_cache: Dict[str, float] = {}

        for lbfgs_step in range(lbfgs_steps):
            def closure() -> torch.Tensor:  # noqa: E306
                lbfgs_optimizer.zero_grad(set_to_none=True)
                residual_l, _ = compute_pde_residual(model, pde, lbfgs_batches["colloc"])
                l_pde = (residual_l**2).mean()
                l_proj = compute_gpc_projection_loss(
                    residual_l, lbfgs_batches["colloc"]["Z"], gpc_basis
                )
                l_bc = compute_bc_loss(model, pde, lbfgs_batches["bc"])
                l_ic = compute_ic_loss(model, pde, lbfgs_batches["ic"])
                F_gram = model.spatial(
                    lbfgs_batches["colloc"]["x"].squeeze(-1).detach(),
                    lbfgs_batches["colloc"]["y"].squeeze(-1).detach(),
                )
                # Orthogonality reg: optionally disabled during L-BFGS
                lbfgs_w_orth = w_orth if lbfgs_keep_reg else 0.0
                lbfgs_w_fac_orth = w_fac_orth if lbfgs_keep_reg else 0.0
                l_reg = model.regularization(F_gram, lbfgs_w_orth, lbfgs_w_fac_orth)
                l_data = compute_data_loss(model, obs_points)

                l_total = (
                    w_pde * l_pde
                    + w_proj * l_proj
                    + w_bc * l_bc
                    + w_ic * l_ic
                    + w_data * l_data
                    + l_reg
                )

                _log_cache.update(
                    {
                        "total": l_total.item(),
                        "pde": l_pde.item(),
                        "proj": l_proj.item(),
                        "bc": l_bc.item(),
                        "ic": l_ic.item(),
                        "data": l_data.item(),
                        "reg": l_reg.item(),
                    }
                )
                if hasattr(pde, "mu_k"):
                    _log_cache["mu_k"] = pde.mu_k.item()
                    _log_cache["sigma"] = pde.sigma.item()

                l_total.backward()
                return l_total

            try:
                lbfgs_optimizer.step(closure)
            except Exception as e:  # noqa: BLE001
                print(f"[L-BFGS] step {lbfgs_step} error: {e}")
                break

            if use_wandb and wandb_run is not None and _log_cache:
                import wandb
                log_dict_lbfgs = {
                    "step": steps + lbfgs_step,
                    "loss/total": _log_cache["total"],
                    "loss/pde": _log_cache["pde"],
                    "loss/proj": _log_cache["proj"],
                    "loss/bc": _log_cache["bc"],
                    "loss/ic": _log_cache["ic"],
                    "loss/data": _log_cache.get("data", 0.0),
                    "loss/reg": _log_cache["reg"],
                }
                if "mu_k" in _log_cache:
                    log_dict_lbfgs["param/mu_k"] = _log_cache["mu_k"]
                    log_dict_lbfgs["param/sigma"] = _log_cache["sigma"]
                    log_dict_lbfgs["error/mu_k_err"] = abs(_log_cache["mu_k"] - pde.true_mu_k)
                    log_dict_lbfgs["error/sigma_err"] = abs(_log_cache["sigma"] - pde.true_sigma)
                wandb.log(log_dict_lbfgs, step=steps + lbfgs_step)

            if lbfgs_step % 10 == 0 and _log_cache:
                param_str = ""
                if "mu_k" in _log_cache:
                    param_str = f" mu_k={_log_cache['mu_k']:.4f} sigma={_log_cache['sigma']:.4f}"
                reg_str = f"{_log_cache['reg']:.4e}" + ("" if lbfgs_keep_reg else " (off)")
                print(
                    f"[L-BFGS {lbfgs_step:5d}/{lbfgs_steps}] "
                    f"total={_log_cache['total']:.4e} "
                    f"pde={_log_cache['pde']:.4e} "
                    f"data={_log_cache['data']:.4e} "
                    f"reg={reg_str}"
                    f"{param_str}"
                )
        print(f"L-BFGS phase complete.")

    final_metrics = evaluate(model, pde, cfg["eval"], device, compute_dtype)
    print("\n=== Final Evaluation ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6e}")

    if use_wandb and wandb_run is not None:
        import wandb  # type: ignore[import-untyped]

        wandb.log({f"final/{k}": v for k, v in final_metrics.items()}, step=steps)
        wandb.log({f"best/{k}": v for k, v in best_metrics.items()}, step=steps)
        wandb.finish()

    ckpt_dir = os.path.join(cfg["logging"].get("log_dir", "checkpoints"), "ckpt")
    if cfg["eval"].get("save_checkpoint", True):
        os.makedirs(ckpt_dir, exist_ok=True)
        pde_type = cfg["pde"].get("type", "diffusion")
        R_ckpt = cfg["model"]["stochastic"]["R"]
        p_ckpt = cfg["model"]["stochastic"]["p"]
        rank_ckpt = cfg["model"]["cp"]["rank"]
        w_proj_ckpt = cfg["training"]["w_proj"]
        ckpt_path = os.path.join(
            ckpt_dir,
            f"model_{pde_type}_R{R_ckpt}_p{p_ckpt}_rank{rank_ckpt}"
            f"_wproj{w_proj_ckpt}_seed{seed}.pt",
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
    import argparse

    parser = argparse.ArgumentParser(description="s2PINN training")
    parser.add_argument("--config", type=str, default="configs/base.yaml")

    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--Mt", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lbfgs_steps", type=int, default=None, help="Additional L-BFGS steps after Adam")
    parser.add_argument("--lbfgs_keep_reg", action="store_true", help="Keep orthogonality regularization during L-BFGS phase")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--colloc", type=int, default=None)
    parser.add_argument("--bc", type=int, default=None)
    parser.add_argument("--ic", type=int, default=None)
    parser.add_argument("--w_pde", type=float, default=None)
    parser.add_argument("--w_proj", type=float, default=None)
    parser.add_argument("--w_bc", type=float, default=None)
    parser.add_argument("--w_ic", type=float, default=None)
    parser.add_argument("--w_orth", type=float, default=None)
    parser.add_argument("--w_fac_orth", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--amp", type=str, default=None, choices=["bf16", "fp16", "none"]
    )
    parser.add_argument("--resample_every", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--uq", action="store_true", help="Enable UQ evaluation")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--w_data", type=float, default=None, help="Data loss weight")
    parser.add_argument("--param_lr_mult", type=float, default=None, help="LR multiplier for PDE params vs model weights")
    parser.add_argument("--init_mu_k", type=float, default=None, help="Initial guess for mu_k")
    parser.add_argument("--init_sigma", type=float, default=None, help="Initial guess for sigma")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.R is not None:
        cfg["model"]["stochastic"]["R"] = args.R
        cfg["pde"]["kl"]["n_terms"] = args.R
    if args.p is not None:
        cfg["model"]["stochastic"]["p"] = args.p
    if args.rank is not None:
        cfg["model"]["cp"]["rank"] = args.rank
    if args.M is not None:
        cfg["model"]["spatial"]["n_atoms"] = args.M
    if args.Mt is not None:
        n_freq = (args.Mt - 1) // 2
        cfg["model"]["temporal"]["n_freq"] = n_freq
    if args.steps is not None:
        cfg["training"]["steps"] = args.steps
    if args.lbfgs_steps is not None:
        cfg["training"]["lbfgs_steps"] = args.lbfgs_steps
    if args.lbfgs_keep_reg:
        cfg["training"]["lbfgs_keep_reg"] = True
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.colloc is not None:
        cfg["training"]["colloc"] = args.colloc
    if args.bc is not None:
        cfg["training"]["bc"] = args.bc
    if args.ic is not None:
        cfg["training"]["ic"] = args.ic
    if args.w_pde is not None:
        cfg["training"]["w_pde"] = args.w_pde
    if args.w_proj is not None:
        cfg["training"]["w_proj"] = args.w_proj
    if args.w_bc is not None:
        cfg["training"]["w_bc"] = args.w_bc
    if args.w_ic is not None:
        cfg["training"]["w_ic"] = args.w_ic
    if args.w_orth is not None:
        cfg["training"]["w_orth"] = args.w_orth
    if args.w_fac_orth is not None:
        cfg["training"]["w_fac_orth"] = args.w_fac_orth
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.amp is not None:
        cfg["training"]["amp"] = args.amp
    if args.resample_every is not None:
        cfg["training"]["resample_every"] = args.resample_every
    if args.gamma is not None:
        cfg.setdefault("pde", {}).setdefault("darcy", {})["gamma"] = args.gamma
    if args.uq:
        cfg.setdefault("eval", {}).setdefault("uq", {})["enabled"] = True
    if args.no_wandb:
        cfg["logging"]["use_wandb"] = False
    if args.w_data is not None:
        cfg["training"]["w_data"] = args.w_data
    if args.param_lr_mult is not None:
        cfg["training"]["param_lr_mult"] = args.param_lr_mult
    if args.init_mu_k is not None:
        cfg["pde"]["init_mu_k"] = args.init_mu_k
    if args.init_sigma is not None:
        cfg["pde"]["init_sigma"] = args.init_sigma

    train(cfg)


if __name__ == "__main__":
    main()
