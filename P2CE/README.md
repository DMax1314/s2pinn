# P2CE: Physics-Informed Polynomial Chaos Baseline

This folder implements a reproduction of the Physics-Informed Polynomial Chaos
(PC2) baseline through UQpy. It is designed as a fair comparison against
S2-PINN, not as a generic PCE toy example.

## Fair Comparison Protocol

- Main paper comparison uses `R=5, p=3`, matching the S2-PINN stochastic dimension and gPC order.
- Sensitivity runs sweep `p in {2, 3, 4}` with seeds `{42, 123, 456}`.
- Model selection for the sensitivity table uses validation PDE/BC residual, not test error.
- PC2 uses physics and boundary/initial conditions only. It does not receive supervised interior solution data.
- Benchmarks:
  - `diffusion_r5`: manufactured transient stochastic diffusion, exact moment and calibration evaluation.
  - `poisson_mc_r5`: non-manufactured stochastic source Poisson, evaluated against an FDM Monte Carlo reference.
  - `darcy_mc_r5`: non-manufactured stochastic Darcy flow, evaluated against an FDM Monte Carlo reference.

Nonlinear Burgers/Allen-Cahn and inverse identification are intentionally
excluded from this PC2 reproduction because the UQpy PC2 interface used here
solves linear constrained PCE coefficient systems. Adding inverse
identification would require a custom outer optimization loop and would no
longer be a direct UQpy PC2 baseline.

## Execution

```bash
cd P2CE
python -m p2ce.runner --config configs/diffusion_r5.yaml
```

Run the non-manufactured benchmarks by switching configs:

```bash
python -m p2ce.runner --config configs/poisson_mc_r5.yaml
python -m p2ce.runner --config configs/darcy_mc_r5.yaml
```

Solver choice, polynomial degree, and seed are controlled by the YAML config.

## Outputs

- Per-run metrics: `P2CE/results/pc2_*.json` and `P2CE/results/pc2_*.csv`.
- Fitted PCE pickles are disabled by default because UQpy distribution objects are not reliably pickleable in this environment.
- Aggregated tables can be generated from the per-run metrics after reproduction.
