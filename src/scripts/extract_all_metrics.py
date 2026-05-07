#!/usr/bin/env python3
"""Extract final UQ metrics from ALL experiment log files on cluster.

Outputs a clean CSV-format table with all methods x PDEs x seeds.
Run on cluster: python3 ~/s2PINN/scripts/extract_all_metrics.py
"""

import os
import glob
import re
import sys

LOG_DIR = os.path.expanduser("~/s2PINN/logs")

# Metrics to extract
UQ_KEYS = [
    "uq/mean_rel_l2",
    "uq/var_rel_l2",
    "uq/coverage_50",
    "uq/coverage_80",
    "uq/coverage_90",
    "uq/coverage_95",
    "uq/sharpness_90",
    "uq/calibration_error",
    "uq/coverage",  # old format (single level)
]
MC_KEYS = [
    "mc_uq/mean_rel_l2",
    "mc_uq/var_rel_l2",
    "mc_uq/coverage",
    "mc_uq/sharpness",
    "mc_uq/coverage_50",
    "mc_uq/coverage_80",
    "mc_uq/coverage_90",
    "mc_uq/coverage_95",
    "mc_uq/calibration_error",
]
ALL_KEYS = UQ_KEYS + MC_KEYS


def extract_metrics(filepath):
    """Extract the LAST occurrence of each metric from a log file."""
    metrics = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                for key in ALL_KEYS:
                    if line.startswith(key + ":"):
                        val = line.split(":", 1)[1].strip()
                        metrics[key] = val
    except Exception as e:
        pass
    return metrics


def classify_file(fname):
    """Classify a log file into method, pde, seed."""
    base = fname.replace(".out", "")
    # Remove SLURM job ID suffix (e.g., _12345678)
    base = re.sub(r"_\d{7,}$", "", base)

    method = None
    pde = None
    seed = None

    # MC-dropout: mcdrop_{pde}_R5_s{seed}
    m = re.match(r"mcdrop_(diff|ac|burg|darcy)_R5_s(\d+)", base)
    if m:
        return "MC-Dropout", m.group(1), m.group(2)

    # PI-DeepONet: don_{pde}_R5_s{seed}
    m = re.match(r"don_(diff|ac|burg|darcy)_R5_s(\d+)", base)
    if m:
        return "PI-DeepONet", m.group(1), m.group(2)

    # Vanilla PINN + proj: vpinn_proj_{pde}_R5_s{seed}
    m = re.match(r"vpinn_proj_(diff|ac|burg|darcy)_R5_s(\d+)", base)
    if m:
        return "VPINN+proj", m.group(1), m.group(2)

    # Vanilla PINN: vpinn_{pde}_R5_s{seed}
    m = re.match(r"vpinn_(diff|ac|burg|darcy)_R5_s(\d+)", base)
    if m:
        return "VPINN", m.group(1), m.group(2)

    # S2-PINN new format proj: s2p_{pde}_proj_s{seed}
    m = re.match(r"s2p_(diff|ac|burg|darcy)_proj_s(\d+)", base)
    if m:
        return "S2PINN-proj", m.group(1), m.group(2)

    # S2-PINN new format nopr: s2p_{pde}_nopr_s{seed}
    m = re.match(r"s2p_(diff|ac|burg|darcy)_nopr_s(\d+)", base)
    if m:
        return "S2PINN-nopr", m.group(1), m.group(2)

    # S2-PINN old format proj: {pde}_R5_proj_uq_s{seed}
    m = re.match(r"(diff|ac|burg)_R5_proj_uq_s(\d+)", base)
    if m:
        return "S2PINN-proj", m.group(1), m.group(2)

    # S2-PINN old format nopr: {pde}_R5_nopr_uq_s{seed}
    m = re.match(r"(diff|ac|burg)_R5_nopr_uq_s(\d+)", base)
    if m:
        return "S2PINN-nopr", m.group(1), m.group(2)

    # Darcy old format: darcy_R5_proj_s{seed} or darcy_R5_nopr_s{seed}
    m = re.match(r"darcy_R5_proj_s(\d+)", base)
    if m:
        return "S2PINN-proj", "darcy", m.group(1)

    m = re.match(r"darcy_R5_nopr_s(\d+)", base)
    if m:
        return "S2PINN-nopr", "darcy", m.group(1)

    return None, None, None


def main():
    # Find all .out files
    all_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.out")))

    results = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        method, pde, seed = classify_file(fname)
        if method is None:
            continue

        metrics = extract_metrics(fpath)
        if not metrics:
            continue

        results.append(
            {"file": fname, "method": method, "pde": pde, "seed": seed, **metrics}
        )

    # De-duplicate: if same (method, pde, seed) appears multiple times, keep the one with more metrics
    seen = {}
    for r in results:
        key = (r["method"], r["pde"], r["seed"])
        if key not in seen or len(r) > len(seen[key]):
            seen[key] = r

    results = sorted(seen.values(), key=lambda x: (x["method"], x["pde"], x["seed"]))

    # Print structured output
    print("=" * 120)
    print("COMPLETE METRICS EXTRACTION")
    print("=" * 120)

    # Group by method
    from collections import defaultdict

    by_method = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    for method in [
        "S2PINN-proj",
        "S2PINN-nopr",
        "VPINN",
        "VPINN+proj",
        "PI-DeepONet",
        "MC-Dropout",
    ]:
        if method not in by_method:
            print(f"\n--- {method}: NO DATA ---")
            continue

        print(f"\n{'=' * 80}")
        print(f"METHOD: {method}")
        print(f"{'=' * 80}")

        for r in sorted(by_method[method], key=lambda x: (x["pde"], x["seed"])):
            print(f"\n  PDE={r['pde']}, Seed={r['seed']}, File={r['file']}")
            for k in ALL_KEYS:
                if k in r:
                    print(f"    {k}: {r[k]}")

    # Print CSV-style summary for easy parsing
    print("\n\n" + "=" * 120)
    print(
        "CSV SUMMARY (method,pde,seed,mean_rel_l2,var_rel_l2,cov50,cov80,cov90,cov95,sharp90,cal_err)"
    )
    print("=" * 120)

    for r in results:
        mean = r.get("uq/mean_rel_l2", r.get("mc_uq/mean_rel_l2", "N/A"))
        var = r.get("uq/var_rel_l2", r.get("mc_uq/var_rel_l2", "N/A"))
        c50 = r.get("uq/coverage_50", r.get("mc_uq/coverage_50", "N/A"))
        c80 = r.get("uq/coverage_80", r.get("mc_uq/coverage_80", "N/A"))
        c90 = r.get(
            "uq/coverage_90",
            r.get(
                "mc_uq/coverage_90",
                r.get("uq/coverage", r.get("mc_uq/coverage", "N/A")),
            ),
        )
        c95 = r.get("uq/coverage_95", r.get("mc_uq/coverage_95", "N/A"))
        s90 = r.get("uq/sharpness_90", r.get("mc_uq/sharpness", "N/A"))
        cal = r.get("uq/calibration_error", r.get("mc_uq/calibration_error", "N/A"))
        print(
            f"{r['method']},{r['pde']},{r['seed']},{mean},{var},{c50},{c80},{c90},{c95},{s90},{cal}"
        )


if __name__ == "__main__":
    main()
