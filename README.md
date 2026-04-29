# LIMIX Modified

Modified CPU-based multivariate GWAS using NumPy, built on [LIMIX](https://github.com/limix/limix).

## Features

- **Multivariate GWAS** — Test for common, any-effect, or phenotype-specific genetic associations
- **Multiple test types** — `common`, `any`, `specific`, `any_vs_common`, `specific_vs_common`
- **Simulation support** — Run batch simulations with effect size scaling (eta) or heterogeneity
- **Multiple testing correction** — Bonferroni, FDR, Holm, Simes-Hochberg

## Installation

**Linux / Windows:**

```bash
git clone https://github.com/bi-horn/LIMIX_modified.git
cd LIMIX_modified
conda env create -f environment.yml -y
conda activate limix_modified
pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 limix-plot>=0.1.2 -e .
```

**macOS (Apple Silicon):** use `environment_macos.yml` instead — Intel MKL is not available on `osx-arm64`, so this file uses Apple's Accelerate framework as the BLAS backend.

```bash
git clone https://github.com/bi-horn/LIMIX_modified.git
cd LIMIX_modified
conda env create -f environment_macos.yml -y
conda activate limix_modified
pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 limix-plot>=0.1.2 -e .
```

> **Why conda?** On Linux/Windows the environment pins NumPy against Intel MKL, which gives deterministic linear algebra (`MKL_CBWR=COMPATIBLE`) and faster BLAS/LAPACK on Intel/AMD CPUs. On macOS (Apple Silicon), NumPy links against Accelerate, which is Apple's optimized BLAS for M-series chips. The `--no-deps` flag keeps pip from replacing the conda-installed NumPy with an OpenBLAS build. See [`install_and_run_notes.md`](install_and_run_notes.md) for verification steps and HPC tuning notes.
## Quick Start

With no arguments, the pipeline runs a demo simulation on an *A. thaliana* Horton dataset (MAF filtered at 0.10):

```bash
limix_modified
```

No config file or arguments needed — the built-in defaults are loaded automatically and results are saved to `./results`.

## Command-Line Interface

All settings have defaults in the built-in config file. You do not need to replace or copy this file. Instead, override individual parameters directly via command-line arguments — any argument you pass overwrites the corresponding value in the config.

```bash
# View all available arguments
limix_modified --help
```

### Overriding defaults

Pass only the parameters you want to change. Everything else keeps its default value:

```bash
# Change the test type and output directory
limix_modified --test_type specific_vs_common --output_directory ./my_results

# Run a specific trait test
limix_modified --test_type specific --pheno_idx 2

```

### Using your own data

Adjust the paths to point to your own genotype and phenotype files:

```bash
limix_modified --geno_path /path/to/genotypes --pheno_path /path/to/phenotypes --dset my_study
```

The `--dset` name is used to organize output files under `results/<dset>/`. When both `--geno_path` and `--pheno_path` are provided, simulated mode is automatically disabled.

### Running simulations

```bash
# Basic simulation with effect size scaling (eta: rescaling proportionality factor)
limix_modified --simulated --eta 0.5 --verbose

# Batch simulation: 10 replicates
limix_modified --simulated --eta 0.5 --n_reps 10

# Resume or extend a batch from a specific replicate
limix_modified --simulated --eta 0.5 --n_reps 1 --start_rep_idx 100

# Heterogeneity simulation mode
limix_modified --simulated --use_heterogeneity --corr_bounds 0.5
```

### Using a custom config file

If you prefer to define your own base configuration, you can point to a custom YAML file. CLI arguments still override any values set in that file:

```bash
limix_modified --config path/to/my_config.yaml --test_type any --verbose
```

### Argument Reference

#### Analysis

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | — | Path to a custom YAML config file. Optional — an internal default config is used when omitted. CLI arguments override values from either config. |
| `--analysis` | `gwas` | Analysis type |
| `--test_type` | `any_vs_common` | Hypothesis test: `common`, `any`, `specific`, `any_vs_common`, `specific_vs_common` |
| `--pheno_idx` | `0` | Trait index for phenotype-specific tests (0-indexed) |
| `--rank` | `4` | Model rank. Recommended to set equal to the number of phenotypes (full rank). |
| `--original_L_init` | `False` | If false a QR-based initialization is used (recommended), if true an all-ones initialization for the genetic covariance matrix is used (not recommended). |

#### Simulation

| Argument | Default | Description |
|----------|---------|-------------|
| `--simulated` | `False` | Flag to enable simulated data mode |
| `--eta` | `0.0` | Proportionality factor for rescaling effect sizes across traits |
| `--rep_idx` | — | Single replicate index (defaults to `0` via config) |
| `--n_reps` | — | Number of replicates for batch run |
| `--start_rep_idx` | `0` | Starting replicate index |
| `--use_heterogeneity` | `False` | If true, heterogeneity GxE instead of rescaling simulation mode is used |
| `--corr_bounds` | `1` | Correlation bounds for heterogeneity simulation |
| `--num_samples` | `250` | Number of samples |
| `--num_tasks` | `4` | Number of traits/tasks for simulation |
| `--ncausal` | `1` | Number of causal variants per trait (only relevant for rescaling effect) |
| `--reference_trait` | `0` | Trait on which to simulate the SNP effect (only relevant for rescaling effect) |

#### Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--dset` | `thaliana_horton` | Dataset name (also used for output directory structure) |
| `--geno_path` | — | Path to genotype data. Pre-configured for the bundled `thaliana_horton` dataset; must be provided for custom datasets. |
| `--pheno_path` | — | Path to phenotype data. Pre-configured for the bundled `thaliana_horton` dataset; must be provided for custom datasets. |
| `--seed` | `42` | Random seed |
| `--transformation_method` | `int` | Phenotype transformation: `int`, `z_score`, `none` |

#### Multiple Testing

| Argument | Default | Description |
|----------|---------|-------------|
| `--correction_method` | `bonferroni` | Correction method: `bonferroni`, `fdr_bh`, `holm`, `simes-hochberg` |
| `--alpha` | `0.05` | Significance threshold |

#### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_directory` | `./results` | Results output directory |
| `--verbose` | `False` | Enable verbose output |

## Configuration File

The built-in default config uses `${PACKAGE_ROOT}` placeholders that resolve automatically at runtime:

```yaml
project: "limix_numpy"
analysis: gwas
test_type: any_vs_common
rank: 4
original_L_init: False
verbose: True
output_directory: ./results

data_param:
  dset: thaliana_horton
  seed: 42
  simulated: False
  eta: 0.0
  num_samples: 250
  num_tasks: 4
  ncausal: 1
  reference_trait: 0
  transformation_method: int
  geno_path: ${PACKAGE_ROOT}/data/genotypes
  ...
```

You only need a custom config file if you want to change settings that are not exposed as CLI arguments. In that case, copy the default and edit the relevant fields — CLI arguments will still override anything in your custom file.

## Output Files

For a real data run, results are stored under the dataset name:

```
results/
└── thaliana_horton/
    ├── log_likelihoods.csv        # LML values and p-values per SNP
    ├── beta_results.csv          # Effect size estimates
    └── gwas_summary.csv          # Summary statistics
```

For batch simulations, the directory structure is organized by simulation parameters:

```
results/
└── eta0.50/
    ├── rep0000/
    │   ├── log_likelihoods.parquet
    │   ├── beta_results.csv
    │   ├── sim_params.csv
    │   └── gwas_summary.csv
    ├── rep0001/
    │   └── ...
    └── ...
```

## Test Types

| Test | Null (H0) | Alternative | Use case |
|------|-----------|-------------|----------|
| `common` | No effect | Same effect across all traits | Shared genetic architecture |
| `any` | No effect | Independent effects per trait | Any genetic signal |
| `specific` | No effect | Common + phenotype-specific effect | Trait-specific signals |
| `any_vs_common` | Common effect | Heterogeneous effects | Detect effect heterogeneity |
| `specific_vs_common` | Common effect | Additional phenotype-specific effect | Single trait deviation |

## Attribution

Based on:

- **LIMIX** (Apache 2.0) — C. Lippert, D. Horta, F. P. Casale, O. Stegle
- **GLIMIX-core** (MIT) — D. Horta

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Author

- Bibiana M. Horn, Christoph Lippert