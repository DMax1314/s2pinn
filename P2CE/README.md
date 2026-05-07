# P2CE: Physics-Informed Polynomial Chaos Baseline

This folder implements a cloud-only reproduction of Lukáš Novák et al.'s Physics-Informed Polynomial Chaos / PC² baseline through UQpy. It is designed as a fair comparison against S²-PINN, not as a generic PCE toy example.

## Fair Comparison Protocol

- Main paper comparison uses `R=5, p=3`, matching the S²-PINN stochastic dimension and gPC order.
- Sensitivity runs sweep `p in {2, 3, 4}` with seeds `{42, 123, 456}`.
- Model selection for the sensitivity table uses validation PDE/BC residual, not test error.
- PC² uses physics and boundary/initial conditions only. It does not receive supervised interior solution data.
- Benchmarks:
  - `diffusion_r5`: manufactured transient stochastic diffusion, exact moment and calibration evaluation.
  - `poisson_mc_r5`: non-manufactured stochastic source Poisson, evaluated against existing 10,000-sample FDM MC reference.
  - `darcy_mc_r5`: non-manufactured stochastic Darcy flow, evaluated against existing 10,000-sample FDM MC reference.

Nonlinear Burgers/Allen-Cahn and inverse identification are intentionally excluded from the PC² v1 reproduction because the UQpy PC² interface used here solves linear constrained PCE coefficient systems. Adding inverse identification would require a custom outer optimization loop and would no longer be a direct UQpy PC² baseline.

## Cloud-Only Execution

Do not run local smoke tests for this folder. Submit to Sol:

```bash
cd ~/s2PINN
bash P2CE/scripts/submit_hawkmem.sh all
```

Useful variants:

```bash
bash P2CE/scripts/submit_hawkmem.sh sanity
bash P2CE/scripts/submit_hawkmem.sh formal
P2CE_SOLVERS=ols bash P2CE/scripts/submit_hawkmem.sh all
```

The default formal array runs both `ols` and `lar`. Use `P2CE_SOLVERS=ols` if LAR is unstable or too expensive.

## Outputs

- Per-run metrics: `P2CE/results/pc2_*.json` and `P2CE/results/pc2_*.csv`
- Fitted PCE pickles are disabled by default because UQpy distribution objects are not reliably pickleable in this environment.
- Aggregated tables:
  - `P2CE/results/tables/all_runs.csv`
  - `P2CE/results/tables/main_p3_summary_by_problem_solver.csv`
  - `P2CE/results/tables/selected_by_validation_summary.csv`

## Notes

The SLURM script installs `UQpy==4.2.0` inside the `s2PINN` conda environment during the remote sanity job. All jobs run on `hawkmem` with CPU BLAS threads configured through `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.
