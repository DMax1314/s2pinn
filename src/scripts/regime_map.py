import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunResult:
    pde: str
    R: int
    proj: int
    seed: int
    metrics: Dict[str, float]


def _parse_seed(name: str) -> Optional[int]:
    m = re.search(r"_s(\d+)", name)
    return int(m.group(1)) if m else None


def _parse_R(name: str) -> Optional[int]:
    m = re.search(r"_R(\d+)", name)
    return int(m.group(1)) if m else None


def _parse_pde_and_cfg(
    filename: str,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    base = os.path.basename(filename)
    proj = 0 if "noproj" in base else 1

    if base.startswith("s2p_R"):
        pde = "diffusion"
        m = re.match(r"s2p_R(\d+)_", base)
        r_dim = int(m.group(1)) if m else None
        return pde, r_dim, proj

    if base.startswith("s2p_AC"):
        pde = "allen_cahn"
        r_dim = _parse_R(base) or 3
        return pde, r_dim, proj

    if base.startswith("s2p_BG"):
        pde = "burgers"
        r_dim = _parse_R(base) or 3
        return pde, r_dim, proj

    return None, None, None


def _parse_final_metrics(text: str) -> Dict[str, float]:
    if "=== Final Evaluation" not in text:
        return {}
    tail = text.split("=== Final Evaluation")[1]
    out: Dict[str, float] = {}
    for line in tail.splitlines():
        m = re.match(r"\s*([\w\d\._]+):\s*([\d\.eE\-\+]+)", line)
        if not m:
            continue
        out[m.group(1)] = float(m.group(2))
    return out


def parse_log_file(path: str) -> Optional[RunResult]:
    pde, R, proj = _parse_pde_and_cfg(path)
    if pde is None or R is None or proj is None:
        return None

    seed = _parse_seed(os.path.basename(path))
    if seed is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    metrics = _parse_final_metrics(txt)
    if not metrics:
        return None
    return RunResult(pde=pde, R=R, proj=proj, seed=seed, metrics=metrics)


def _mean(xs: Iterable[float]) -> float:
    xs_list = list(xs)
    if not xs_list:
        raise ValueError("empty")
    return sum(xs_list) / float(len(xs_list))


def _std(xs: Iterable[float]) -> float:
    xs_list = list(xs)
    if len(xs_list) <= 1:
        return 0.0
    mu = _mean(xs_list)
    var = sum((x - mu) ** 2 for x in xs_list) / float(len(xs_list) - 1)
    return math.sqrt(var)


def metric_t1_avg_over_z(metrics: Dict[str, float]) -> Optional[float]:
    keys = ["eval_Z0_t1.0", "eval_Z_e1_t1.0", "eval_Z_e2_t1.0"]
    vals = [metrics.get(k) for k in keys if k in metrics]
    vals_f = [v for v in vals if v is not None]
    if not vals_f:
        return None
    return float(sum(vals_f) / len(vals_f))


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    import csv

    if not rows:
        raise ValueError("no rows")
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_regime_map(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate regime_map.png; install it in the environment"
        ) from e

    pdes = ["diffusion", "allen_cahn", "burgers"]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.6), sharey=True)
    if len(pdes) == 1:
        axes = [axes]

    for ax, pde in zip(axes, pdes):
        sub = [r for r in summary_rows if r["pde"] == pde]
        for proj, label, color in [
            (1, "with proj", "#1f77b4"),
            (0, "no proj", "#d62728"),
        ]:
            rows = [r for r in sub if int(r["proj"]) == proj]
            rows_sorted = sorted(rows, key=lambda r: int(r["R"]))
            xs = [int(r["R"]) for r in rows_sorted]
            ys = [float(r["metric_mean"]) for r in rows_sorted]
            yerr = [float(r["metric_std"]) for r in rows_sorted]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2)
            ax.fill_between(
                xs,
                [max(y - e, 1e-12) for y, e in zip(ys, yerr)],
                [y + e for y, e in zip(ys, yerr)],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        ax.set_title(pde.replace("_", " "))
        ax.set_xlabel("R")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Rel L2 @ t=1 (avg over z_eval)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    log_dir = os.path.abspath(log_dir)
    paths = sorted(glob.glob(os.path.join(log_dir, "*.out")))
    runs: List[RunResult] = []
    for p in paths:
        r = parse_log_file(p)
        if r is not None:
            runs.append(r)

    if not runs:
        raise SystemExit(f"No parsable runs found in {log_dir}")

    raw_rows: List[Dict[str, Any]] = []
    for r in runs:
        m = metric_t1_avg_over_z(r.metrics)
        if m is None:
            continue
        raw_rows.append(
            {
                "pde": r.pde,
                "R": r.R,
                "proj": r.proj,
                "seed": r.seed,
                "metric": m,
            }
        )

    out_csv_raw = os.path.join(os.getcwd(), "regime_map_raw.csv")
    write_csv(raw_rows, out_csv_raw)

    groups: Dict[Tuple[str, int, int], List[float]] = {}
    for row in raw_rows:
        key = (str(row["pde"]), int(row["R"]), int(row["proj"]))
        groups.setdefault(key, []).append(float(row["metric"]))

    summary_rows: List[Dict[str, Any]] = []
    for (pde, r_dim, proj), vals in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        summary_rows.append(
            {
                "pde": pde,
                "R": r_dim,
                "proj": proj,
                "n": len(vals),
                "metric_mean": _mean(vals),
                "metric_std": _std(vals),
            }
        )

    out_csv = os.path.join(os.getcwd(), "regime_map_summary.csv")
    write_csv(summary_rows, out_csv)

    out_png = os.path.join(os.getcwd(), "regime_map.png")
    plot_regime_map(summary_rows, out_png)

    print(f"Wrote {out_csv_raw}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
