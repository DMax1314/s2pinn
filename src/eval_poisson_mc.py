import argparse

import torch

from spinn.train import build_model, build_pde, load_config


def _build_vanilla_model(R: int, hidden_dim: int, n_layers: int, device: torch.device):
    """Build VanillaPINN without importing at module level."""
    from spinn.models.vanilla_pinn import VanillaPINN

    model = VanillaPINN(
        input_dim=3 + R,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="tanh",
        dropout=0.0,
    )
    return model.to(device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mean_ref", type=str, required=True)
    parser.add_argument("--var_ref", type=str, required=True)
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=262144)
    parser.add_argument(
        "--model_type", type=str, default="s2pinn",
        choices=["s2pinn", "vanilla"],
        help="Model type: s2pinn (default) or vanilla PINN baseline",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    cfg = load_config(args.config)
    cfg["pde"]["type"] = "poisson_mc"
    cfg["model"]["stochastic"]["R"] = args.R
    cfg["pde"]["kl"]["n_terms"] = args.R

    pde = build_pde(cfg)

    if args.model_type == "vanilla":
        model = _build_vanilla_model(args.R, args.hidden_dim, args.n_layers, device)
    else:
        model = build_model(cfg, device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded {args.model_type} checkpoint from {args.checkpoint}")

    mean_ref = torch.load(args.mean_ref, map_location=device)
    var_ref = torch.load(args.var_ref, map_location=device)
    grid_res = mean_ref.shape[0]

    x_lin = torch.linspace(-1.0, 1.0, grid_res, device=device, dtype=dtype)
    y_lin = torch.linspace(-1.0, 1.0, grid_res, device=device, dtype=dtype)
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x_flat = x_grid.reshape(-1, 1)
    y_flat = y_grid.reshape(-1, 1)

    B = x_flat.shape[0]
    t_flat = torch.full((B, 1), 1.0, device=device, dtype=dtype)

    Z = pde.sample_z(args.N, device=device, dtype=dtype)
    nz = Z.shape[0]

    t_rep = t_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    x_rep = x_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    y_rep = y_flat.unsqueeze(1).expand(B, nz, 1).reshape(-1, 1)
    Z_rep = Z.unsqueeze(0).expand(B, nz, -1).reshape(-1, Z.shape[1])

    u_pred_all = []
    total = t_rep.shape[0]
    for s in range(0, total, args.batch):
        e = min(s + args.batch, total)
        u_pred_all.append(model(t_rep[s:e], x_rep[s:e], y_rep[s:e], Z_rep[s:e]))

    u_pred = torch.cat(u_pred_all, dim=0).reshape(B, nz)

    mean_pred = u_pred.mean(dim=1).reshape(grid_res, grid_res)
    var_pred = u_pred.var(dim=1, unbiased=False).reshape(grid_res, grid_res)

    def rel_l2(pred, ref):
        err = torch.sqrt(((pred - ref) ** 2).sum())
        norm = torch.sqrt((ref**2).sum()).clamp_min(1e-12)
        return (err / norm).item()

    mean_err = rel_l2(mean_pred, mean_ref)
    var_err = rel_l2(var_pred, var_ref)

    print("\n=== RESULTS ===")
    print(f"Model type:             {args.model_type}")
    print(f"Mean Relative L2 Error: {mean_err:.4e}")
    print(f"Var Relative L2 Error:  {var_err:.4e}")

    res_dict = {
        "mean_pred": mean_pred.cpu(),
        "var_pred": var_pred.cpu(),
        "mean_ref": mean_ref.cpu(),
        "var_ref": var_ref.cpu(),
        "x": x_grid.cpu(),
        "y": y_grid.cpu(),
    }
    out_name = f"poisson_mc_{args.model_type}_predictions.pt"
    torch.save(res_dict, out_name)
    print(f"Saved fields to {out_name}")

if __name__ == "__main__":
    main()
