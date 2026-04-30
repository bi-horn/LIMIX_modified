import os
import sys
import yaml
import argparse
import traceback
from typing import Dict, Any, Optional
import uuid
import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests as mt
from limix_modified.run_numpylimix._numpy_dataloader import get_data_numpy
from pathlib import Path
from importlib import resources

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


def get_package_root() -> Path:
    """Get the root directory of the run_numpylimix package."""
    return Path(resources.files("limix_modified"))

def get_default_config_path():
    """Get default config path from package resources."""
    try:
        config_path = resources.files("limix_modified.run_numpylimix.configs") / "hyperparameters.yaml"
        config_str = str(config_path)
        if os.path.exists(config_str):
            return config_str
        else:
            print(f"[WARNING] Default config not found at: {config_str}")
            return None
    except Exception as e:
        print(f"[WARNING] Could not load default config: {e}")
        return None
    
def resolve_path_placeholders(path: Optional[str]) -> Optional[str]:
    """Replace ${PACKAGE_ROOT} placeholder with actual package root path."""
    if path is None:
        return None
    package_root = str(get_package_root())
    resolved = path.replace("${PACKAGE_ROOT}", package_root)
    resolved = os.path.expanduser(resolved)
    return resolved


def resolve_config_text(config_text: str) -> str:
    """Replace ${PACKAGE_ROOT} placeholder in config file content."""
    package_root = str(get_package_root())
    return config_text.replace("${PACKAGE_ROOT}", package_root)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LIMIX (CPU/NumPy-based) multivariate GWAS analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to the config file"
    )
    # Analysis
    parser.add_argument("--analysis", type=str, choices=["gwas"])
    parser.add_argument("--test_type", type=str,
                        choices=["common", "any", "specific", "specific_vs_common", "any_vs_common"])
    parser.add_argument("--pheno_idx", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--original_L_init", action="store_true")
    
    # Simulation
    parser.add_argument("--eta", type=float)
    parser.add_argument("--rep_idx", type=int)
    parser.add_argument("--n_reps", type=int)
    parser.add_argument("--start_rep_idx", type=int, default=0)
    parser.add_argument("--use_heterogeneity", action="store_true")
    parser.add_argument("--corr_bounds", type=float)
    
    # Data
    parser.add_argument("--dset", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--ncausal", type=int)
    parser.add_argument("--reference_trait", type=int)
    parser.add_argument("--transformation_method", type=str, choices=["int", "z_score", "none"])

    parser.add_argument("--geno_path", type=str, help="Path to genotype data")
    parser.add_argument("--pheno_path", type=str, help="Path to phenotype data")

    # Output
    parser.add_argument("--output_directory", type=str)
    parser.add_argument("--verbose", action="store_true")
    
    # Multiple testing
    parser.add_argument("--correction_method", type=str, default="bonferroni",
                        choices=["bonferroni", "fdr_bh", "holm", "simes-hochberg"])
    parser.add_argument("--alpha", type=float, default=0.05)
    
    return parser.parse_args()


def load_config(config_path: str, args) -> Dict[str, Any]:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config_text = f.read()
    
    config = yaml.safe_load(resolve_config_text(config_text))
    
    # Top-level overrides
    for key in ['analysis', 'test_type', 'pheno_idx', 'rank', 'output_directory']:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    
    if args.verbose:
        config['verbose'] = True

    if args.original_L_init:
        config['original_L_init'] = True

    # Data param overrides
    data_param = config.setdefault('data_param', {})
    for key in ['dset', 'seed', 'num_samples', 'num_tasks', 'ncausal', 'reference_trait', 'transformation_method', 'geno_path', 'pheno_path']:
        if getattr(args, key, None) is not None:
            data_param[key] = getattr(args, key)
    
    if args.rep_idx is not None:
        data_param['rep_idx'] = args.rep_idx   
    if args.eta is not None:
        data_param['eta'] = args.eta
    if args.corr_bounds is not None:
        data_param['corr_bounds'] = args.corr_bounds
    if args.simulated:
        data_param['simulated'] = True
    if args.use_heterogeneity:
        data_param['use_heterogeneity'] = True
    
    # Store at top level for convenience
    config['rep_idx'] = data_param.get('rep_idx')
    config['eta'] = data_param.get('eta')
    config['corr_bounds'] = data_param.get('corr_bounds')
    config['use_heterogeneity'] = data_param.get('use_heterogeneity', False)

    # After all data_param overrides, add:
    if data_param.get('geno_path') and data_param.get('pheno_path'):
        data_param['simulated'] = False

    return config

def setup_test_matrices(test_type: str, p: int, pheno_idx: int = 0):
    """Setup A0/A1 matrices for LIMIX testing."""
    if test_type == "common":
        A0, A1 = np.ones((p, 1)), None
    elif test_type == "any":
        A0, A1 = np.eye(p), None
    elif test_type == "specific":
        A0 = np.zeros((p, 2))
        A0[:, 0] = 1.0
        A0[pheno_idx, 1] = 1.0
        A1 = None
    elif test_type == "specific_vs_common":
        A0 = np.ones((p, 1))
        A1 = np.zeros((p, 1))
        A1[pheno_idx, 0] = 1.0
    elif test_type == "any_vs_common":
        A0 = np.ones((p, 1))
        A1 = np.eye(p)[:, :-1]
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    return A0, A1


def run_gwas(config: Dict[str, Any], rep_idx: Optional[int] = None,
             correction_method: str = "bonferroni", alpha: float = 0.05) -> Dict[str, Any]:
    """Run GWAS analysis."""
    from limix_modified.qtl._scan import scan

    
    verbose = config.get("verbose", True)
    simulated = config.get("data_param", {}).get("simulated")

    eta = config.get("eta") if config.get("eta") is not None else config.get("data_param", {}).get("eta")
    if rep_idx is None:
        rep_idx = config.get("rep_idx") if config.get("rep_idx") is not None else config.get("data_param", {}).get("rep_idx")
    
    if simulated: 
        print(f"[INFO] Running GWAS: eta={eta}, rep_idx={rep_idx}")
    else:
        print(f"[INFO] Running GWAS.")
    
    # Load data
    G, K, Y, meta = get_data_numpy(config, rep_idx, eta)
    print(f"[INFO] Data: G={G.shape}, Y={Y.shape}, K={K.shape}")
    
    # Setup matrices
    p = Y.shape[1]
    A = np.eye(p)
    test_type = config.get("test_type", "any_vs_common")
    pheno_idx = config.get("pheno_idx", 0)
    rank = config.get("rank", 1)
    A0, A1 = setup_test_matrices(test_type, p, pheno_idx)

    print(f"[INFO] original_L_init={config.get('original_L_init', False)}, "
        f"rank={config.get('rank', 1)}, test_type={test_type}")
        
    # Run scan
    M = np.ones((Y.shape[0], 1))
    qtl_results = scan(
        G, Y, lik="normal", K=K, M=M, A=A, A0=A0, A1=A1, rank=rank,
        original_L_init=config.get("original_L_init", False),
        verbose=verbose
    )
    
    stats = qtl_results.stats
    
    # Multiple testing correction
    pv_col = "pv20" if "pv20" in stats.columns else "pv10"
    reject, pvals_corrected, alpha_sidak, alpha_bonf = mt(
        stats[pv_col].values, alpha=alpha, method=correction_method
    )
    
    results = {
        "qtl_results": qtl_results,
        "stats": stats,
        "reject": reject,
        "pvals_corrected": pvals_corrected,
        "alpha_bonferroni": alpha_bonf,
        "n_significant": np.sum(reject),
        "simulation_info": meta.get('simulation_info'),
        "config": config,
        "eta": eta,          
        "rep_idx": rep_idx,  
    }
    
    print(f"[INFO] Significant hits: {np.sum(reject)} (α={alpha}, {correction_method})")
    return results


def save_results(results: Dict[str, Any]) -> str:
    """Save results to disk."""
    config = results.get("config", {})
    data_param = config.get("data_param", {})
    simulated = data_param.get("simulated", False)

    rep_idx = results.get("rep_idx")
    if rep_idx is None:
        rep_idx = config.get("rep_idx")
    if rep_idx is None:
        rep_idx = data_param.get("rep_idx")

    eta = results.get("eta")
    if eta is None:
        eta = config.get("eta")
    if eta is None:
        eta = data_param.get("eta")

    output_dir = config.get("output_directory", "./results")
    use_heterogeneity = data_param.get("use_heterogeneity", False)
    corr_bounds = data_param.get("corr_bounds")
    test_type = config.get("test_type", "any_vs_common")

    # Build output path
    if simulated and rep_idx is not None:
        if use_heterogeneity and corr_bounds is not None:
            sim_folder = f"corr{corr_bounds:.0f}"
        elif eta is not None:
            eta_str = f"{abs(eta):.2f}"
            sim_folder = f"eta-{eta_str}" if eta < 0 else f"eta{eta_str}"
        else:
            sim_folder = "simulation"
        base_dir = os.path.join(output_dir, sim_folder, f"rep{rep_idx:04d}")
    else:
        base_dir = os.path.join(output_dir, data_param.get("dset", "unknown"))

    os.makedirs(base_dir, exist_ok=True)

    has_h2 = test_type in ["specific_vs_common", "any_vs_common"]

    # Log likelihoods
    stats = results.get("stats")
    if stats is not None:
        if has_h2:
            headers = [
                "snp_index", "lml0", "lml1", "lml2",
                "lrt10", "df10", "pv10",
                "lrt20", "df20", "pv20",
                "lrt21", "df21", "pv12"
            ]
        else:
            headers = [
                "snp_index", "lml0", "lml1",
                "lrt10", "df10", "pv10"
            ]

        if "scale_H0" in stats.columns:
            headers.extend(["scale_H0", "scale_H1"])
            if has_h2:
                headers.append("scale_H2")

        likelihood_df = stats[headers].copy() if all(h in stats.columns for h in headers) else stats.copy()

        if simulated:
            likelihood_path = os.path.join(base_dir, "log_likelihoods.parquet")
            likelihood_df.to_parquet(likelihood_path, index=False)
        else:
            likelihood_path = os.path.join(base_dir, "log_likelihoods.csv")
            likelihood_df.to_csv(likelihood_path, index=False)

        print(f"[INFO] Saved likelihoods to: {likelihood_path}")
        
    # Beta results
    qtl_results = results.get("qtl_results")
    if qtl_results is not None and hasattr(qtl_results, 'effsizes'):
        effsizes = qtl_results.effsizes
        pivot_dfs = []

        for hypothesis, df in effsizes.items():
            betas = df[df["effect_type"] == "candidate"].copy()

            # Pivot so each env becomes a column pair (effsize + se)
            for i, env_name in enumerate(betas["env"].unique()):
                env_rows = betas[betas["env"] == env_name]
                col_prefix = f"{hypothesis}_beta{i}"
                pivot = env_rows[["test", "effsize", "effsize_se"]].rename(columns={
                    "test": "snp_index",
                    "effsize": col_prefix,
                    "effsize_se": f"{col_prefix}_se",
                })
                pivot_dfs.append(pivot.set_index("snp_index"))

        beta_combined = pd.concat(pivot_dfs, axis=1).reset_index()

        if simulated:
            beta_path = os.path.join(base_dir, "beta_results.parquet")
            beta_combined.to_parquet(beta_path, index=False)
        else:
            beta_path = os.path.join(base_dir, "beta_results.csv")
            beta_combined.to_csv(beta_path, index=False)

        print(f"[INFO] Saved betas to: {beta_path}")

    # Simulation parameters (simulated only)
    if simulated:
        sim_params_path = os.path.join(base_dir, "sim_params.csv")
        sim_info = results.get("simulation_info") or {}

        if use_heterogeneity:
            n_traits = sim_info.get('n_traits', 2)
            context_indices = sim_info.get('heterogeneity_context_indices', [])

            context_headers = [f"context{i}_indices" for i in range(n_traits)]
            if rep_idx is not None:
                headers = ["rep_idx", "use_heterogeneity", "corr_bounds", "n_traits"] + context_headers + ["ncausal", "rank"]
                row = [rep_idx, use_heterogeneity, corr_bounds, n_traits]
            else:
                headers = ["use_heterogeneity", "n_traits"] + context_headers + ["ncausal", "rank"]
                row = [use_heterogeneity, n_traits]

            for i in range(n_traits):
                if i < len(context_indices) and context_indices[i] is not None:
                    row.append(str(context_indices[i]))
                else:
                    row.append("None")

            row.extend([sim_info.get('ncausal'), config.get('rank', 1)])

        else:
            rescaling_indices = sim_info.get('rescaling_common_indices', [])
            if rep_idx is not None:
                headers = ["rep_idx", "use_heterogeneity", "rescaling_common_indices", "eta", "rank"]
                row = [rep_idx, use_heterogeneity, str(rescaling_indices), eta, config.get('rank', 1)]
            else:
                headers = ["use_heterogeneity", "rescaling_common_indices", "eta", "rank"]
                row = [use_heterogeneity, str(rescaling_indices), eta, config.get('rank', 1)]

        sim_params_df = pd.DataFrame([row], columns=headers)
        sim_params_df.to_csv(sim_params_path, index=False)
        print(f"[INFO] Saved sim params to: {sim_params_path}")

    # GWAS summary
    stats_len = len(stats) if stats is not None else 0
    metrics = ['n_significant', 'alpha_bonferroni', 'total_tests']
    values = [
        results.get('n_significant', 0),
        results.get('alpha_bonferroni', np.nan),
        stats_len,
    ]

    if simulated:
        metrics.extend(['eta', 'rep_idx'])
        values.extend([
            eta if eta is not None else np.nan,
            rep_idx if rep_idx is not None else np.nan,
        ])

    summary = pd.DataFrame({'metric': metrics, 'value': values})
    summary.to_csv(os.path.join(base_dir, "gwas_summary.csv"), index=False)

    print(f"[INFO] All results saved to: {base_dir}")
    return base_dir


def run_single(config: Dict[str, Any], args, rep_idx: Optional[int] = None) -> Dict[str, Any]:
    """Run single analysis."""
    analysis = config.get("analysis", "gwas").lower()
    
    if analysis == "gwas":
        result = run_gwas(config, rep_idx, args.correction_method, args.alpha)
        save_results(result)
        return result
    else:
        raise ValueError(f"Unknown analysis: {analysis}")


def run_batch(config: Dict[str, Any], args) -> list:
    """Run batch of simulations."""
    n_reps = args.n_reps
    start_idx = args.start_rep_idx
    
    print(f"[INFO] Batch run: {n_reps} reps starting at {start_idx}")
    
    all_results = []
    for i in range(n_reps):
        rep_idx = start_idx + i
        print(f"\n[INFO] === Rep {rep_idx} ({i+1}/{n_reps}) ===")
        try:
            result = run_single(config, args, rep_idx)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Rep {rep_idx} failed: {e}")
            traceback.print_exc()
    
    return all_results


def main():
    args = parse_args()
    config = load_config(args.config, args)
    
    limix_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if limix_path not in sys.path:
        sys.path.insert(0, limix_path)
    
    uid = str(uuid.uuid4())[:8]
    print(f"\n[INFO] Run ID: {uid}")
    print(f"[INFO] Analysis: {config.get('analysis', 'gwas')}")
    print(f"[INFO] Dataset: {config.get('data_param', {}).get('dset', 'unknown')}")
    print(f"[INFO] Seed: {config.get('data_param', {}).get('seed', 42)}")
    
    if args.n_reps is not None:
        results = run_batch(config, args)
    else:
        results = run_single(config, args, args.rep_idx)
    
    print("[INFO] Done!")
    return results


if __name__ == "__main__":
    main()

