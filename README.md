# LIMIX Modified

Modified CPU-based multivariate GWAS using NumPy, built on [LIMIX](https://github.com/limix/limix).

## Features

- **Multivariate GWAS** — Test for common, any-effect, or trait-specific genetic associations
- **Multiple test types** — `common`, `any`, `specific`, `any_vs_common`, `specific_vs_common`
- **Simulation support** — Run batch simulations with effect size scaling (eta) or heterogeneity
- **Multiple testing correction** — Bonferroni, FDR, Holm, Simes-Hochberg

## Installation

```bash
git clone https://github.com/bi-horn/LIMIX_modified.git
cd LIMIX_modified
conda env create -f environment.yml -y
conda activate limix_modified
pip install --no-deps ndarray-listener numpy-sugar optimix brent-search glimix-core==3.1.14 limix-plot>=0.1.2 -e .
```

> **Why conda?** The environment pins NumPy against Intel MKL, which gives
> deterministic linear algebra (`MKL_CBWR=COMPATIBLE`) and faster BLAS/LAPACK
> on Intel/AMD CPUs. The `--no-deps` flag keeps pip from replacing MKL-linked
> NumPy with an OpenBLAS build. See [`test_install.md`](test_install.md) for
> verification steps and HPC tuning notes.

## Quick Start

With no arguments, the pipeline runs a demo simulation on an *A. thaliana* Horton dataset (MAF filtered at 0.10):

```bash
limix-modified
```

No config file or arguments needed — the built-in defaults are loaded automatically and results are saved to `./results`.

## Command-Line Interface

All settings have sensible defaults in the built-in config file. You do **not** need to replace or copy this file. Instead, override individual parameters directly via command-line arguments — any argument you pass takes precedence over the corresponding value in the config.

```bash
# View all available arguments
limix-modified --help
```

### Overriding defaults

Pass only the parameters you want to change. Everything else keeps its default value:

```bash
# Change the test type and output directory
limix-modified --test_type specific_vs_common --output_directory ./my_results

# Run a specific trait test
limix-modified --test_type specific --pheno_idx 2

# Adjust multiple testing correction
limix-modified --correction_method fdr_bh --alpha 0.01
```

### Using your own data

Point the pipeline to your own genotype and phenotype files:

```bash
limix-modified --geno_path /path/to/genotypes --pheno_path /path/to/phenotypes --dset my_study
```

The `--dset` name is used to organize output files under `results/<dset>/`.

### Running simulations

```bash
# Basic simulation with effect size scaling
limix_modified --simulated --eta 0.5 --verbose

# Batch simulation: 100 replicates
limix_modified --simulated --eta 0.5 --n_reps 100

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
| `--config` | built-in | Path to a custom YAML config file. Optional — the built-in default is used when omitted. CLI arguments override values from either config. |
| `--analysis` | `gwas` | Analysis type |
| `--test_type` | `any_vs_common` | Hypothesis test: `common`, `any`, `specific`, `any_vs_common`, `specific_vs_common` |
| `--pheno_idx` | `0` | Trait index for trait-specific tests (0-indexed) |
| `--rank` | `1` | Model rank |

#### Simulation

| Argument | Default | Description |
|----------|---------|-------------|
| `--simulated` | off | Flag to enable simulated data mode |
| `--eta` | — | Effect size scaling parameter |
| `--rep_idx` | — | Single replicate index |
| `--n_reps` | — | Number of replicates for batch run |
| `--start_rep_idx` | `0` | Starting replicate index |
| `--use_heterogeneity` | off | Use heterogeneity simulation mode |
| `--corr_bounds` | — | Correlation bounds for heterogeneity simulation |

#### Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--dset` | `thaliana_horton` | Dataset name (also used for output directory structure) |
| `--geno_path` | built-in | Path to genotype data |
| `--pheno_path` | built-in | Path to phenotype data |
| `--seed` | `42` | Random seed |
| `--num_samples` | — | Number of samples |
| `--num_tasks` | — | Number of traits/tasks |
| `--ncausal` | — | Number of causal variants |
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
| `--verbose` | off | Enable verbose output |

## Configuration File

The built-in default config uses `${PACKAGE_ROOT}` placeholders that resolve automatically at runtime:

```yaml
analysis: gwas
test_type: any_vs_common
rank: 1
output_directory: ./results

data_param:
  dset: thaliana_horton
  seed: 42
  geno_path: ${PACKAGE_ROOT}/data/genotypes
  pheno_path: ${PACKAGE_ROOT}/data/phenotypes
  transformation_method: int
```

You only need a custom config file if you want to change settings that are not exposed as CLI arguments. In that case, copy the default and edit the relevant fields — CLI arguments will still override anything in your custom file.

## Output Files

For a real data run, results are stored under the dataset name:

```
results/
└── thaliana_horton/
    ├── log_likelihoods.csv    # LML values and p-values per SNP
    ├── beta_results.csv       # Effect size estimates
    └── gwas_summary.csv       # Summary statistics
```

For batch simulations, the directory structure is organized by simulation parameters:

```
results/
└── eta0.50/
    ├── rep0000/
    │   ├── log_likelihoods.csv
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
| `specific` | No effect | Common + trait-specific effect | Trait-specific signals |
| `any_vs_common` | Common effect | Heterogeneous effects | Detect effect heterogeneity |
| `specific_vs_common` | Common effect | Additional trait-specific effect | Single trait deviation |

## Attribution

Based on:

- **LIMIX** (Apache 2.0) — C. Lippert, D. Horta, F. P. Casale, O. Stegle
- **GLIMIX-core** (MIT) — D. Horta

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Author

- Bibiana M. Horn, Christoph Lippert