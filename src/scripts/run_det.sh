#!/bin/bash
# Deterministic S-PINN experiments for five PDEs

set -e

# Base config
CONFIG="configs/base.yaml"
LOGDIR="logs/det"

# Diffusion 2D
python experiments.py --config $CONFIG --pde diffusion2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 0 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde diffusion2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 1 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde diffusion2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 2 --log_dir $LOGDIR

# Helmholtz 3D
python experiments.py --config $CONFIG --pde helmholtz3d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 0 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde helmholtz3d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 1 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde helmholtz3d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 2 --log_dir $LOGDIR

# Klein–Gordon 2D
python experiments.py --config $CONFIG --pde kg_2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 0 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde kg_2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 1 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde kg_2d --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 2 --log_dir $LOGDIR

# Navier–Stokes 3D (100k steps)
python experiments.py --config $CONFIG --pde ns3d_mms --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 0 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde ns3d_mms --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 1 --log_dir $LOGDIR
python experiments.py --config $CONFIG --pde ns3d_mms --coeff cp --rank 16 --R 0 --p 0 --quadrature tensor --seed 2 --log_dir $LOGDIR
