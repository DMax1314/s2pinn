# S2-PINN Code

[Zhendong Li*](https://github.com/DMax1314),[Alwum Onwunta](https://sites.google.com/view/akwumonwunta/home)

*Corresponding author.

Conference on Neural Information Processing Systems (NeurIPS 2026)

[Openreview](https://openreview.net/forum?id=lwTb5ix4w9), [ArXiv](https://arxiv.org/abs/2610.03303)

---
This directory contains the code used for the S2-PINN experiment


- `src/spinn/`: S2-PINN models, stochastic PDE definitions, gPC utilities, and training loops.
- `src/configs/`: YAML configurations for manufactured, non-manufactured, and inverse benchmarks.
- `src/*.py`: training and evaluation entry points for S2-PINN and neural baselines.
- `P2CE/`: physics-informed PCE baseline implementation and configs.


## Environment

Install all dependencies, including the FEniCSx/MPI stack required by the
reference solvers:

```bash
mamba create -n s2pinn -c conda-forge python=3.10 fenics-dolfinx=0.9 mpi4py petsc4py slepc4py
mamba activate s2pinn
pip install -r requirements.txt
pip install -r requirements-fenics.txt
```

The FEniCSx/MPI stack is required for the reference solvers in
`src/solvers/fenics/`:

- `dolfinx` / FEniCSx, tested with dolfinx 0.9
- `ufl`
- `mpi4py`
- PETSc/SLEPc runtime libraries provided by the FEniCSx installation

Verify the installation:

```bash
python -c "import torch, dolfinx, mpi4py, petsc4py, slepc4py; print('environment ok')"
```

## Basic Usage

Run a representative S2-PINN training job:

```bash
python src/spinn/train.py --config src/configs/diffusion.yaml --seed 42
```

Run the PC2 baseline:

```bash
cd P2CE
pip install -r requirements.txt
python -m p2ce.runner --config configs/diffusion_r5.yaml
```


---
## BibTeX
```bibtex
@misc{li2026s2pinn,
      title={S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks},
      author={Zhendong Li and Akwum Onwunta},
      year={2026},
      eprint={2610.03303},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.03303},
}
```