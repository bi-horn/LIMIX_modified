# Installation verification

```bash
# Conda installation used for simulation runs and likelihood evaluation because
# conda can install numpy linked against Intel MKL, providing:
#   - Deterministic linear algebra via MKL_CBWR=COMPATIBLE (not available with OpenBLAS)
#   - Bitwise reproducible results regardless of CPU core count or hardware
#   - Multi-threaded BLAS/LAPACK performance on Intel/AMD CPUs
# Using pip alone would install numpy with OpenBLAS (or reference BLAS),
# which is slower on Intel hardware and has no deterministic mode.

# 1. Create env from file
cd LIMIX_modified/
conda env create -f environment.yml -y
conda activate limix_modified

# 2. Install LIMIX and all its proprietary dependencies without breaking MKL
pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 limix-plot>=0.1.2 -e .

# 3. Verify import
python -c "import limix_modified; print('Import: OK')"

# 4. Verify MKL is correctly linked to numpy
python -c "import numpy as np; np.__config__.show()"

# 5. Clean up
conda deactivate
conda env remove -n limix_modified -y
```

## Reproducible mode

For numerical reproducibility across different CPU architectures (e.g. when
validating results against TorchLIMIX), set:

```bash
export MKL_CBWR=COMPATIBLE
```

This forces MKL to produce bitwise identical results regardless of the
underlying hardware. Useful when comparing outputs across login nodes,
compute nodes, or different clusters.

> **Note:** `MKL_CBWR=COMPATIBLE` can significantly reduce performance and is
> not recommended for scalability runs.

## Performance tuning (SLURM)

For batch simulations, match thread counts to your allocation so MKL and
OpenMP use all available cores:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
```