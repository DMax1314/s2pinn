#!/usr/bin/env python
"""Aggregate PC^2 run JSON files into paper-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


def read_rows(root: Path) -> List[Dict]:
    rows = []
    for path in sorted(root.glob("pc2_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = dict(payload["metrics"])
        row["json_path"] = str(path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict], group_keys: List[str]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    metric_keys = [
        "mean_rel_l2",
        "var_rel_l2",
        "validation_error",
        "fit_error",
        "elapsed_seconds",
        "basis_count",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "calibration_error",
        "sharpness_90",
    ]
    summaries = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = {key: value for key, value in zip(group_keys, group)}
        summary["n_runs"] = len(group_rows)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group_rows if metric in row and row[metric] != ""]
            if not values:
                continue
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def select_by_validation(rows: List[Dict]) -> List[Dict]:
    best = {}
    for row in rows:
        key = (row.get("basis_type", ""), row["problem"], row["solver"], row["seed"])
        current = best.get(key)
        if current is None or float(row["validation_error"]) < float(current["validation_error"]):
            best[key] = row
    return list(best.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="P2CE/results", help="Directory containing pc2_*.json files.")
    parser.add_argument("--out_dir", default="P2CE/results/tables", help="Output table directory.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    rows = read_rows(root)
    write_csv(out_dir / "all_runs.csv", rows)
    write_csv(out_dir / "main_p3_runs.csv", [row for row in rows if int(row["p"]) == 3])
    write_csv(out_dir / "summary_by_problem_p_solver.csv", summarize(rows, ["problem", "p", "solver"]))
    write_csv(
        out_dir / "main_p3_summary_by_problem_solver.csv",
        summarize([row for row in rows if int(row["p"]) == 3], ["problem", "solver"]),
    )
    if any("p_stochastic" in row and "p_deterministic" in row for row in rows):
        pz3_rows = [row for row in rows if int(row.get("p_stochastic", -1)) == 3]
        write_csv(out_dir / "main_pz3_runs.csv", pz3_rows)
        write_csv(
            out_dir / "summary_by_problem_pz_pdet_solver.csv",
            summarize(rows, ["problem", "p_stochastic", "p_deterministic", "solver"]),
        )
        write_csv(
            out_dir / "main_pz3_summary_by_problem_pdet_solver.csv",
            summarize(pz3_rows, ["problem", "p_deterministic", "solver"]),
        )
    validation_selected = select_by_validation(rows)
    write_csv(out_dir / "selected_by_validation_runs.csv", validation_selected)
    write_csv(
        out_dir / "selected_by_validation_summary.csv",
        summarize(validation_selected, ["problem", "solver"]),
    )
    print(f"Aggregated {len(rows)} PC^2 runs into {out_dir}")


if __name__ == "__main__":
    main()
