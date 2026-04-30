# Installation verification

The setup is slightly different on Mac vs. Linux/Windows because Intel MKL only exists on x86 (Linux/Windows). On Apple Silicon Macs, NumPy uses Apple's Accelerate framework instead.

## Linux / Windows

We use conda here so NumPy gets linked against Intel MKL. This gives you faster BLAS/LAPACK on Intel/AMD CPUs. Installing with pip alone would pull in OpenBLAS, which is slower.

```bash
cd LIMIX_modified/

# 1. Create the environment
conda env create -f environment.yml -y
conda activate limix_modified

# 2. Install LIMIX and its dependencies (--no-deps keeps MKL-linked NumPy intact)
pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 "limix-plot>=0.1.2" -e .

# 3. Check the install
python -c "import limix_modified; print('Import: OK')"

# 4. Check that NumPy is using MKL (look for 'mkl' in the output)
python -c "import numpy as np; np.show_config()"
```

To remove the environment later (if wanted):

```bash
conda deactivate
conda env remove -n limix_modified -y
```

## macOS (Apple Silicon)

Use `environment_macos.yml` instead — it skips MKL and lets NumPy use Accelerate. Everything else is the same.

```bash
cd LIMIX_modified/

conda env create -f environment_macos.yml -y
conda activate limix_modified

pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 "limix-plot>=0.1.2" -e .

python -c "import limix_modified; print('Import: OK')"
python -c "import numpy as np; np.show_config()"   # should mention 'accelerate'
```

> Note: in zsh and bash, the quotes around `"limix-plot>=0.1.2"` are needed — otherwise the shell reads `>=` as a redirect and the install fails.


## Reproducible mode

For bitwise-identical results across different machines (e.g. when validating against TorchLIMIX):

```bash
export MKL_CBWR=COMPATIBLE
```

This makes MKL produce the same numbers regardless of the CPU but it's noticeably slower, so don't use it for scalability benchmarks.

On macOS this flag does nothing (no MKL). If you need cross-platform reproducibility, run the reference simulations on Linux/Windows with `MKL_CBWR=COMPATIBLE` and compare against those.

## Performance tuning (SLURM)

For batch jobs, match the thread counts to your allocation:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
```