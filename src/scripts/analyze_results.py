import glob
import os
import re
import csv
import math
from collections import defaultdict


def parse_log_file(filepath):
    """Parses a single log file to extract metrics and config."""
    filename = os.path.basename(filepath)
    # Format: jobname_jobid.out
    # Job name format: s2p_default_s42 or bl_h128_L4_s42

    parts = filename.split("_")
    # Try to extract job ID (last part)
    try:
        job_id = parts[-1].replace(".out", "")
        job_name_base = "_".join(parts[:-1])
    except:
        return None

    # Extract seed if present (s42)
    seed_match = re.search(r"_s(\d+)", job_name_base)
    seed = int(seed_match.group(1)) if seed_match else None

    # Extract experiment type
    exp_type = "unknown"
    if "s2p" in job_name_base:
        if "default" in job_name_base:
            exp_type = "default"
        elif "no_proj" in job_name_base:
            exp_type = "no_proj"
        elif "no_gram" in job_name_base:
            exp_type = "no_gram"
        elif "no_fac" in job_name_base:
            exp_type = "no_fac_orth"
        elif "no_reg" in job_name_base:
            exp_type = "no_reg"
        elif "elliptic" in job_name_base:
            exp_type = "elliptic"
        elif "rank" in job_name_base:
            m = re.search(r"rank(\d+)", job_name_base)
            if m:
                exp_type = f"rank{m.group(1)}"
        elif "p" in job_name_base and "s2p_p" in job_name_base:
            m = re.search(r"_p(\d+)", job_name_base)
            if m:
                exp_type = f"p{m.group(1)}"
        elif "M" in job_name_base:
            m = re.search(r"_M(\d+)", job_name_base)
            if m:
                exp_type = f"M{m.group(1)}"
        elif "R" in job_name_base:
            m = re.search(r"_R(\d+)", job_name_base)
            if m:
                exp_type = f"R{m.group(1)}"
    elif "bl" in job_name_base:
        h = re.search(r"_h(\d+)", job_name_base)
        L = re.search(r"_L(\d+)", job_name_base)
        if h and L:
            exp_type = f"baseline_h{h.group(1)}_L{L.group(1)}"

    metrics = {}
    with open(filepath, "r") as f:
        content = f.read()

        # Look for Final Evaluation block
        if "=== Final Evaluation" in content:
            # Extract all metrics from the final block
            final_block = content.split("=== Final Evaluation")[-1]
            for line in final_block.split("\n"):
                match = re.search(r"\s+([\w\d\._]+):\s+([\d\.eE\-\+]+)", line)
                if match:
                    key = match.group(1)
                    val = float(match.group(2))
                    metrics[key] = val
        else:
            # Try to find the last [EVAL] line
            eval_lines = re.findall(r"\[EVAL step=\d+\] (.*)", content)
            if eval_lines:
                last_line = eval_lines[-1]
                parts = last_line.split()
                for part in parts:
                    if "=" in part:
                        try:
                            k, v = part.split("=")
                            metrics[k] = float(v)
                        except:
                            pass
            else:
                return None  # No results found

    if not metrics:
        return None

    res = {
        "job_id": job_id,
        "job_name": job_name_base,
        "exp_type": exp_type,
        "seed": seed,
    }
    res.update(metrics)
    return res


def main():
    log_dir = "../logs"
    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} not found. Run from src/")
        return

    log_files = glob.glob(os.path.join(log_dir, "*.out"))
    data = []

    print(f"Found {len(log_files)} log files.")

    # Collect all metric keys
    all_metric_keys = set()

    for f in log_files:
        res = parse_log_file(f)
        if res:
            data.append(res)
            for k in res.keys():
                if k.startswith("eval_"):
                    all_metric_keys.add(k)

    if not data:
        print("No valid results found in logs.")
        return

    # Group by exp_type
    grouped = defaultdict(list)
    for row in data:
        grouped[row["exp_type"]].append(row)

    # Calculate stats
    summary = []
    header = ["exp_type", "count"] + sorted(list(all_metric_keys))

    # Prepare summary data structure for easy plotting later
    summary_dict = {}

    print("\n=== Results Summary ===")

    # We want mean and std for each metric
    for exp_type, rows in grouped.items():
        n = len(rows)
        row_summary = {"exp_type": exp_type, "count": n}

        print(f"\nExperiment: {exp_type} (n={n})")

        for metric in all_metric_keys:
            vals = [r.get(metric) for r in rows if r.get(metric) is not None]
            if not vals:
                row_summary[f"{metric}_mean"] = None
                row_summary[f"{metric}_std"] = None
                continue

            mean_val = sum(vals) / len(vals)
            variance = (
                sum((x - mean_val) ** 2 for x in vals) / len(vals)
                if len(vals) > 0
                else 0
            )
            # Use sample std dev if n > 1
            if len(vals) > 1:
                variance = sum((x - mean_val) ** 2 for x in vals) / (len(vals) - 1)
            std_val = math.sqrt(variance)

            row_summary[f"{metric}_mean"] = mean_val
            row_summary[f"{metric}_std"] = std_val

            # Print key metrics
            if "t1.0" in metric or "t0.5" in metric:
                print(f"  {metric}: {mean_val:.2e} +/- {std_val:.2e}")

        summary.append(row_summary)

    # Save summary to CSV
    summary_keys = ["exp_type", "count"]
    for k in sorted(list(all_metric_keys)):
        summary_keys.append(f"{k}_mean")
        summary_keys.append(f"{k}_std")

    with open("results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(summary)

    # Save raw data
    raw_keys = ["job_id", "job_name", "exp_type", "seed"] + sorted(
        list(all_metric_keys)
    )
    with open("results_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_keys)
        writer.writeheader()
        writer.writerows(data)

    print("\nSaved to results_raw.csv and results_summary.csv")

    # Generate LaTeX table
    print("\n=== LaTeX Table (Main) ===")
    main_exps = ["default", "baseline_h128_L4", "baseline_h256_L6"]

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & $t=0.5$ (Mean) & $t=1.0$ (Mean) & Params \\")
    print(r"\midrule")

    for exp in main_exps:
        # Find the summary row
        row = next((r for r in summary if r["exp_type"] == exp), None)
        if not row:
            continue

        # Calculate average error across Z points for t=0.5
        t05_metrics = [k for k in all_metric_keys if "t0.5" in k]
        t10_metrics = [k for k in all_metric_keys if "t1.0" in k]

        if not t05_metrics or not t10_metrics:
            continue

        # Aggregated mean of means
        val_t05 = sum(row[f"{k}_mean"] for k in t05_metrics) / len(t05_metrics)
        # Approximate pooled std? Just avg std for now
        std_t05 = sum(row[f"{k}_std"] for k in t05_metrics) / len(t05_metrics)

        val_t10 = sum(row[f"{k}_mean"] for k in t10_metrics) / len(t10_metrics)
        std_t10 = sum(row[f"{k}_std"] for k in t10_metrics) / len(t10_metrics)

        name = exp.replace("_", " ").title()
        if "Default" in name:
            name = "s$^2$PINN (Ours)"
        if "Baseline" in name:
            name = name.replace("Baseline", "PINN")

        print(
            f"{name} & {val_t05:.2e} $\\pm$ {std_t05:.2e} & {val_t10:.2e} $\\pm$ {std_t10:.2e} & - \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Relative $L_2$ errors (mean $\pm$ std over 3 seeds).}")
    print(r"\label{tab:main_results}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
