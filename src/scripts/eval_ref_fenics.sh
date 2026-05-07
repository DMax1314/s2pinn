#!/bin/bash
# Evaluate FEniCS reference solutions for all PDEs and random configs

set -e

OUTDIR="ref_outputs"
mkdir -p $OUTDIR

# Diffusion 2D deterministic
python solvers/fenics/diffusion2d_ref.py --mesh 32 --t_final 1.0 --num_steps 50 --output $OUTDIR/diff2d_ref.npz

# Helmholtz 3D deterministic
python solvers/fenics/helmholtz3d_ref.py --mesh 16 --k 3.1415926 --output $OUTDIR/helmholtz3d_ref.npz

# Klein–Gordon 2D deterministic
python solvers/fenics/kg_2d_ref.py --mesh 32 --t_final 1.0 --num_steps 50 --m 1.0 --output $OUTDIR/kg_2d_ref.npz

# Navier–Stokes 3D deterministic
python solvers/fenics/ns3d_mms_ref.py --mesh 8 --t_final 1.0 --num_steps 20 --nu 0.01 --output $OUTDIR/ns3d_mms_ref.npz

# Stochastic reference (example for R=3, p=3)
python solvers/fenics/diffusion2d_ref.py --mesh 32 --t_final 1.0 --num_steps 50 --rand_R 3 --rand_p 3 --quadrature tensor --output $OUTDIR/diff2d_ref_R3_p3.npz
python solvers/fenics/helmholtz3d_ref.py --mesh 16 --k 3.1415926 --rand_R 3 --rand_p 3 --quadrature tensor --output $OUTDIR/helmholtz3d_ref_R3_p3.npz
python solvers/fenics/kg_2d_ref.py --mesh 32 --t_final 1.0 --num_steps 50 --m 1.0 --rand_R 3 --rand_p 3 --quadrature tensor --output $OUTDIR/kg_2d_ref_R3_p3.npz
python solvers/fenics/ns3d_mms_ref.py --mesh 8 --t_final 1.0 --num_steps 20 --nu 0.01 --rand_R 3 --rand_p 3 --quadrature tensor --output $OUTDIR/ns3d_mms_ref_R3_p3.npz
