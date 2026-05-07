import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (10, 6),
        }
    )


def load_data():
    if not os.path.exists("results_summary.csv"):
        print("Error: results_summary.csv not found!")
        return None
    return pd.read_csv("results_summary.csv")


def plot_ablation(df):
    # Select ablation experiments
    ablations = ["default", "no_proj", "no_gram", "no_fac_orth", "no_reg"]
    labels = ["Default", "No Proj", "No Gram", "No Fac Orth", "No Reg"]

    # Filter rows
    subset = df[df["exp_type"].isin(ablations)].copy()

    # Reorder
    subset["exp_type"] = pd.Categorical(
        subset["exp_type"], categories=ablations, ordered=True
    )
    subset = subset.sort_values("exp_type")

    # Calculate aggregate metric (average across all Z points at t=1.0)
    cols = [c for c in df.columns if "t1.0_mean" in c]
    subset["mean_error"] = subset[cols].mean(axis=1)

    # Calculate pooled std (approx)
    std_cols = [c for c in df.columns if "t1.0_std" in c]
    subset["std_error"] = subset[std_cols].mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(
        labels,
        subset["mean_error"],
        yerr=subset["std_error"],
        capsize=5,
        alpha=0.8,
        color="skyblue",
    )
    plt.ylabel("Relative L2 Error (t=1.0)")
    plt.title("Ablation Study: Impact of Components")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("../tex/spinn_fig/ablation_study.pdf")
    plt.close()
    print("Saved ablation_study.pdf")


def plot_sweeps(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Rank Sweep
    ranks = [4, 8, 16, 32]  # 16 is default
    rank_exps = ["rank4", "rank8", "default", "rank32"]

    subset = df[df["exp_type"].isin(rank_exps)].copy()
    # Map default to rank16 for plotting
    subset.loc[subset["exp_type"] == "default", "rank"] = 16
    subset.loc[subset["exp_type"] == "rank4", "rank"] = 4
    subset.loc[subset["exp_type"] == "rank8", "rank"] = 8
    subset.loc[subset["exp_type"] == "rank32", "rank"] = 32

    subset = subset.sort_values("rank")

    cols = [c for c in df.columns if "t1.0_mean" in c]
    y = subset[cols].mean(axis=1)
    yerr = subset[[c for c in df.columns if "t1.0_std" in c]].mean(axis=1)

    axes[0].errorbar(subset["rank"], y, yerr=yerr, fmt="-o", capsize=5)
    axes[0].set_xlabel("CP Rank (q)")
    axes[0].set_ylabel("Relative L2 Error")
    axes[0].set_title("Effect of CP Rank")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ranks)
    axes[0].set_xticklabels(ranks)

    # 2. Spatial Basis (M) Sweep
    Ms = [64, 128, 256, 512]  # 256 is default
    M_exps = ["M64", "M128", "default", "M512"]

    subset = df[df["exp_type"].isin(M_exps)].copy()
    subset.loc[subset["exp_type"] == "default", "M"] = 256
    subset.loc[subset["exp_type"] == "M64", "M"] = 64
    subset.loc[subset["exp_type"] == "M128", "M"] = 128
    subset.loc[subset["exp_type"] == "M512", "M"] = 512

    subset = subset.sort_values("M")

    y = subset[cols].mean(axis=1)
    yerr = subset[[c for c in df.columns if "t1.0_std" in c]].mean(axis=1)

    axes[1].errorbar(subset["M"], y, yerr=yerr, fmt="-s", capsize=5, color="orange")
    axes[1].set_xlabel("Spatial Basis Size (M)")
    axes[1].set_title("Effect of Spatial Resolution")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(Ms)
    axes[1].set_xticklabels(Ms)

    # 3. gPC Order (p) Sweep
    ps = [1, 2, 3, 4]  # 3 is default
    p_exps = ["p1", "p2", "default", "p4"]

    subset = df[df["exp_type"].isin(p_exps)].copy()
    subset.loc[subset["exp_type"] == "default", "p"] = 3
    subset.loc[subset["exp_type"] == "p1", "p"] = 1
    subset.loc[subset["exp_type"] == "p2", "p"] = 2
    subset.loc[subset["exp_type"] == "p4", "p"] = 4

    subset = subset.sort_values("p")

    y = subset[cols].mean(axis=1)
    yerr = subset[[c for c in df.columns if "t1.0_std" in c]].mean(axis=1)

    axes[2].errorbar(subset["p"], y, yerr=yerr, fmt="-^", capsize=5, color="green")
    axes[2].set_xlabel("gPC Order (p)")
    axes[2].set_title("Effect of Polynomial Order")
    axes[2].set_xticks(ps)

    plt.tight_layout()
    plt.savefig("../tex/spinn_fig/parameter_sweeps.pdf")
    plt.close()
    print("Saved parameter_sweeps.pdf")


def main():
    set_style()
    df = load_data()
    if df is None:
        return

    os.makedirs("../tex/spinn_fig", exist_ok=True)

    plot_ablation(df)
    plot_sweeps(df)


if __name__ == "__main__":
    main()
