import argparse
import itertools
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from spinn.models.vanilla_pinn import VanillaPINN
from spinn.train import build_pde, load_config, sample_batch, set_seed


def gauss_nodes_weights_1d(n: int, dist: str) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dist == "gaussian":
        x, w = np.polynomial.hermite.hermgauss(n)
        z = np.sqrt(2.0) * x
        w = w / math.sqrt(math.pi)
        return z.astype(np.float64), w.astype(np.float64)
    if dist == "uniform":
        x, w = np.polynomial.legendre.leggauss(n)
        w = w / 2.0
        return x.astype(np.float64), w.astype(np.float64)
    raise ValueError(f"Unsupported dist: {dist}")


def tensor_nodes_weights(
    nodes_1d: np.ndarray, w_1d: np.ndarray, R: int
) -> Tuple[np.ndarray, np.ndarray]:
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    grids = list(itertools.product(range(nodes_1d.shape[0]), repeat=R))
    Z = np.zeros((len(grids), R), dtype=np.float64)
    W = np.ones((len(grids),), dtype=np.float64)
    for i, idxs in enumerate(grids):
        for r, j in enumerate(idxs):
            Z[i, r] = nodes_1d[j]
            W[i] *= w_1d[j]
    return Z, W


def sample_probes(
    n: int, pde: Any, device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    batch = pde.sample_collocation(n, device=device, dtype=dtype)
    return {
        "t": batch["t"].squeeze(-1),
        "x": batch["x"].squeeze(-1),
        "y": batch["y"].squeeze(-1),
    }


@torch.no_grad()
def mc_reference_moments(
    pde: Any,
    probes: Dict[str, torch.Tensor],
    n_mc: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = probes["t"]
    x = probes["x"]
    y = probes["y"]
    B = t.shape[0]
    Z = pde.sample_z(n_mc, device=device, dtype=dtype)

    t_rep = t.unsqueeze(1).expand(B, n_mc).reshape(-1)
    x_rep = x.unsqueeze(1).expand(B, n_mc).reshape(-1)
    y_rep = y.unsqueeze(1).expand(B, n_mc).reshape(-1)
    Z_rep = Z.unsqueeze(0).expand(B, n_mc, -1).reshape(-1, Z.shape[1])

    u = pde.exact_solution(t_rep, x_rep, y_rep, Z_rep).reshape(B, n_mc)
    mean = u.mean(dim=1)
    var = u.var(dim=1, unbiased=False)
    return mean, var


def rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    err = torch.linalg.norm(pred - ref)
    denom = torch.linalg.norm(ref).clamp_min(1e-12)
    return (err / denom).item()


def train_deterministic_pinn(
    cfg: Dict[str, Any],
    pde: Any,
    z_fixed: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> VanillaPINN:
    model = VanillaPINN(
        input_dim=3,
        hidden_dim=int(cfg.get("baseline", {}).get("hidden_dim", 128)),
        n_layers=int(cfg.get("baseline", {}).get("n_layers", 4)),
        activation=str(cfg.get("baseline", {}).get("activation", "tanh")),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    steps = int(cfg["training"]["steps"])
    n_colloc = int(cfg["training"]["colloc"])
    n_bc = int(cfg["training"]["bc"])
    n_ic = int(cfg["training"]["ic"])
    log_every = int(cfg.get("logging", {}).get("log_every", 100))
    resample_every = int(cfg["training"].get("resample_every", 1))

    w_pde = float(cfg["training"]["w_pde"])
    w_bc = float(cfg["training"]["w_bc"])
    w_ic = float(cfg["training"]["w_ic"])

    z_fixed = z_fixed.to(device=device, dtype=dtype).view(1, -1)
    z_empty_cache: Dict[int, torch.Tensor] = {}

    batches = None
    for step in range(steps):
        if batches is None or (resample_every > 0 and step % resample_every == 0):
            batches = sample_batch(pde, n_colloc, n_bc, n_ic, device, dtype)
            for key in ("colloc", "bc", "ic"):
                B = batches[key]["t"].shape[0]
                batches[key]["Z"] = z_fixed.expand(B, -1)

        opt.zero_grad(set_to_none=True)

        col = batches["colloc"]
        t = col["t"].squeeze(-1)
        x = col["x"].squeeze(-1)
        y = col["y"].squeeze(-1)
        Z = col["Z"]

        Bc = t.shape[0]
        z_empty = z_empty_cache.get(Bc)
        if z_empty is None:
            z_empty = torch.empty(Bc, 0, device=device, dtype=dtype)
            z_empty_cache[Bc] = z_empty

        derivs = model.forward_with_derivatives(t, x, y, z_empty)
        residual = pde.compute_residual(t, x, y, Z, derivs)
        loss_pde = (residual**2).mean()

        bc = batches["bc"]
        t_b = bc["t"].squeeze(-1)
        x_b = bc["x"].squeeze(-1)
        y_b = bc["y"].squeeze(-1)
        Z_b = bc["Z"]
        Bb = t_b.shape[0]
        z_empty_b = z_empty_cache.get(Bb)
        if z_empty_b is None:
            z_empty_b = torch.empty(Bb, 0, device=device, dtype=dtype)
            z_empty_cache[Bb] = z_empty_b
        u_bc = model(t_b, x_b, y_b, z_empty_b)
        u_bc_ref = pde.exact_solution(t_b, x_b, y_b, Z_b)
        loss_bc = ((u_bc - u_bc_ref) ** 2).mean()

        ic = batches["ic"]
        t_i = ic["t"].squeeze(-1)
        x_i = ic["x"].squeeze(-1)
        y_i = ic["y"].squeeze(-1)
        Z_i = ic["Z"]
        Bi = t_i.shape[0]
        z_empty_i = z_empty_cache.get(Bi)
        if z_empty_i is None:
            z_empty_i = torch.empty(Bi, 0, device=device, dtype=dtype)
            z_empty_cache[Bi] = z_empty_i
        u_ic = model(t_i, x_i, y_i, z_empty_i)
        u_ic_ref = pde.exact_solution(t_i, x_i, y_i, Z_i)
        loss_ic = ((u_ic - u_ic_ref) ** 2).mean()

        loss = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
        loss.backward()
        opt.step()

        if log_every > 0 and step % log_every == 0:
            print(
                f"[{step:>5d}/{steps}] total={loss.item():.4e} pde={loss_pde.item():.4e} "
                f"bc={loss_bc.item():.4e} ic={loss_ic.item():.4e}"
            )

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--max_nodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--n_probe", type=int, default=2048)
    ap.add_argument("--n_mc_ref", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = int(args.seed)
    set_seed(int(args.seed))

    if args.R is not None:
        cfg["model"]["stochastic"]["R"] = int(args.R)
        cfg["pde"]["kl"]["n_terms"] = int(args.R)
    if args.gamma is not None:
        cfg.setdefault("pde", {}).setdefault("darcy", {})["gamma"] = float(args.gamma)
    if args.steps is not None:
        cfg["training"]["steps"] = int(args.steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    pde = build_pde(cfg)
    pde_type = cfg.get("pde", {}).get("type", "diffusion")
    R = int(cfg["model"]["stochastic"]["R"])
    dist = str(cfg["model"]["stochastic"]["dist"])

    nodes_1d, w_1d = gauss_nodes_weights_1d(int(args.q), dist)
    Z_np, W_np = tensor_nodes_weights(nodes_1d, w_1d, R)
    n_nodes = Z_np.shape[0]
    if n_nodes > int(args.max_nodes):
        raise SystemExit(
            f"Too many tensor nodes: {n_nodes} (R={R}, q={args.q}). "
            f"Increase --max_nodes or reduce q/R."
        )

    Z_nodes = torch.from_numpy(Z_np).to(device=device, dtype=dtype)
    W_t = torch.from_numpy(W_np).to(device=device, dtype=dtype)
    W_t = W_t / W_t.sum().clamp_min(1e-12)

    # WandB setup
    use_wandb = not args.no_wandb and cfg.get("logging", {}).get("use_wandb", True)
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            run_name = f"sc_{pde_type}_R{R}_q{args.q}_seed{args.seed}"
            wandb_run = wandb.init(
                project=cfg.get("logging", {}).get("project", "s2pinn"),
                entity=cfg.get("logging", {}).get("entity"),
                name=run_name,
                config={
                    **cfg,
                    "model_type": "stochastic_collocation",
                    "q": args.q,
                    "n_nodes": n_nodes,
                },
                reinit=True,
            )
        except Exception:
            use_wandb = False

    print(
        f"SC baseline | {pde_type} | nodes: {n_nodes} (R={R}, q={args.q}, dist={dist})"
    )

    probes = sample_probes(int(args.n_probe), pde, device=device, dtype=dtype)
    mean_ref, var_ref = mc_reference_moments(
        pde, probes, int(args.n_mc_ref), device=device, dtype=dtype
    )

    mean_acc = torch.zeros_like(mean_ref)
    m2_acc = torch.zeros_like(mean_ref)

    total_train_time = 0.0
    models: List[VanillaPINN] = []
    for i in range(n_nodes):
        print(f"\nNode {i + 1}/{n_nodes} z={Z_nodes[i].tolist()}")
        t0 = time.time()
        model_i = train_deterministic_pinn(
            cfg, pde, Z_nodes[i], device=device, dtype=dtype
        )
        node_time = time.time() - t0
        total_train_time += node_time
        models.append(model_i)

        Bp = probes["t"].shape[0]
        z_empty = torch.empty(Bp, 0, device=device, dtype=dtype)
        u_pred = model_i(probes["t"], probes["x"], probes["y"], z_empty)
        w_i = W_t[i]
        mean_acc = mean_acc + w_i * u_pred
        m2_acc = m2_acc + w_i * (u_pred**2)

        if use_wandb and wandb_run is not None:
            import wandb  # type: ignore[import-untyped]

            wandb.log(
                {"sc/node_idx": i, "sc/node_time_s": node_time},
                step=i,
            )

    var_acc = (m2_acc - mean_acc**2).clamp_min(0.0)

    mean_err = rel_l2(mean_acc, mean_ref)
    var_err = rel_l2(var_acc, var_ref)
    print(f"\n=== SC Baseline UQ Metrics ({pde_type}) ===")
    print(f"uq/mean_rel_l2: {mean_err:.6e}")
    print(f"uq/var_rel_l2:  {var_err:.6e}")
    print(f"total_train_time_s: {total_train_time:.1f}")

    if use_wandb and wandb_run is not None:
        import wandb  # type: ignore[import-untyped]

        wandb.log(
            {
                "final/uq_mean_rel_l2": mean_err,
                "final/uq_var_rel_l2": var_err,
                "final/total_train_time_s": total_train_time,
                "final/n_nodes": n_nodes,
            },
            step=n_nodes,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
