import argparse
import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinn.train import build_model, build_pde, load_config


def plot_solution(model, pde, t_val, z_val, z_name, grid_res, device, save_dir):
    model.eval()

    # Create grid
    x_lin = torch.linspace(pde.x_range[0], pde.x_range[1], grid_res, device=device)
    y_lin = torch.linspace(pde.y_range[0], pde.y_range[1], grid_res, device=device)
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing="ij")

    x = x_grid.reshape(-1, 1)
    y = y_grid.reshape(-1, 1)
    num_points = x.shape[0]
    t = torch.full((num_points, 1), float(t_val), device=device)

    # Prepare Z
    Z = torch.tensor(z_val, device=device).unsqueeze(0).expand(num_points, -1)

    # Evaluate
    with torch.no_grad():
        u_pred = model(t.squeeze(-1), x.squeeze(-1), y.squeeze(-1), Z)
        u_exact = pde.exact_solution(t, x, y, Z)

    u_pred = u_pred.cpu().numpy().reshape(grid_res, grid_res)
    u_exact = u_exact.cpu().numpy().reshape(grid_res, grid_res)
    error = np.abs(u_pred - u_exact)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Common extent
    extent = [pde.x_range[0], pde.x_range[1], pde.y_range[0], pde.y_range[1]]

    # Prediction
    im0 = axes[0].imshow(u_pred.T, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title(f"Prediction (t={t_val}, {z_name})")
    plt.colorbar(im0, ax=axes[0])

    # Exact
    im1 = axes[1].imshow(u_exact.T, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title(f"Exact (t={t_val}, {z_name})")
    plt.colorbar(im1, ax=axes[1])

    # Error
    im2 = axes[2].imshow(error.T, origin="lower", extent=extent, cmap="inferno")
    axes[2].set_title(f"Abs Error (t={t_val}, {z_name})")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    filename = f"solution_{z_name}_t{t_val}.pdf"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument("--out_dir", type=str, default="tex/spinn_fig")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]

    # Build model and PDE
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pde = build_pde(cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    # Get evaluation points
    t_eval = cfg["eval"]["t_eval"]
    z_eval = cfg["eval"]["z_eval"]
    grid_res = 128  # Higher res for viz

    print(
        f"Generating plots for {len(z_eval)} Z-points and {len(t_eval)} time-points..."
    )

    for z_info in z_eval:
        for t_val in t_eval:
            plot_solution(
                model,
                pde,
                t_val,
                z_info["values"],
                z_info["name"],
                grid_res,
                device,
                args.out_dir,
            )


if __name__ == "__main__":
    main()
