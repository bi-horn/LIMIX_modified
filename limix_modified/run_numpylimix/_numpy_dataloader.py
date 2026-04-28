from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from limix_modified.run_numpylimix._standardize import standardize_data
from pathlib import Path
from importlib import resources
import os
from pathlib import Path
from importlib import resources



@dataclass
class DataPaths:
    """File paths for dataset loading."""
    pheno: Optional[str] = None
    geno: Optional[str] = None
    annot: Optional[str] = None
    batch: Optional[str] = None
    cov: Optional[str] = None
    config: Optional[str] = None


@dataclass
class SimulationConfig:
    """Parameters for phenotype simulation."""
    enabled: bool = False
    num_samples: int = 250
    eta: float = 0.5
    ncausal: int = 1
    corr_bounds: float = 0.0
    use_heterogeneity: bool = False
    num_tasks: int = 2
    reference_trait: Optional[int] = None
    rep_idx: Optional[int] = None

@dataclass
class CorrectionConfig:
    """Phenotype correction settings."""
    regress_batch: bool = True
    regress_covariates: bool = True
    per_trait_batch: bool = False
    transformation: str = "none"


@dataclass
class AnalysisConfig:
    """LIMIX analysis configuration."""
    analysis: str = "gwas"  
    test_type: str = "any_vs_common"
    pheno_idx: int = 0
    rank: int = 1
    original_L_init: bool = False
    mean_type: str = "fixed"
    correction_method: str = "bonferroni"
    alpha: float = 0.05
    verbose: bool = True


@dataclass
class NumpyDataConfig:
    """Complete configuration for numpy data loading."""
    dset: str = "custom"
    seed: int = 42
    paths: DataPaths = field(default_factory=DataPaths)
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    correction_config: CorrectionConfig = field(default_factory=CorrectionConfig)
    root: Optional[str] = None


class NumpyDataLoader:
    """
    NumPy-based data loader matching torch MultitaskDatasetSNP behavior.
    Except that it does not retun the stablized G matrix because the QS decomposition is already implemented in the scan function.
    """
    
    def __init__(self, config: NumpyDataConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
    
        self._rng = np.random.RandomState(config.seed)
        self._log(f"[INFO] Initialized RNG with seed={config.seed}")
        
        self.simulation_info = None
        self.snp_positions = None
        
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load data and return (G, K, Y, meta).
        """
        G_df, snp_positions = self._load_genotype()
        self.snp_positions = snp_positions

        Y_df, simulation_info = self._load_phenotype(G_df)
        self.simulation_info = simulation_info

        # Align while still DataFrames (indices preserved)
        G_aligned, Y_aligned, common_index = self._align_samples(G_df, Y_df)

        G = G_aligned.values.astype(np.float64)
        Y = Y_aligned.values.astype(np.float64)

        Y = self._apply_transformation(Y)
        K = self._compute_kinship(G)

        self.trait_variances = np.var(Y, axis=0)
        self.trait_covariances = np.cov(Y.T) if Y.shape[1] > 1 else np.array([[np.var(Y)]])

        meta = self._build_metadata(common_index)

        self._log(f"[INFO] Data loaded: G={G.shape}, Y={Y.shape}, K={K.shape}")

        return G, K, Y, meta
    
    def _load_genotype(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load genotype data: subsample FIRST, normalize SECOND. Returns DataFrame."""
        
        self._log(f"[INFO] Loading genotype for {self.config.dset}")
        
        # Load raw genotype (DataFrame with index)
        gen_data, bim = self._load_genotype_source()
        snp_positions = self._load_snp_positions(bim)
        
        self._log(f"[INFO] Raw genotype shape: {gen_data.shape}")
        
        if self.config.sim_config.enabled and self.config.dset in ["thaliana_horton", "thaliana_1001"]:
            gen_data, snp_positions = self._subsample_for_simulation(gen_data, snp_positions)
            self._log(f"[INFO] Subsampled to: {gen_data.shape}")
        
        # Normalize but keep as DataFrame
        G_df = self._normalize_genotype(gen_data)
        
        self._log(f"[INFO] Normalized genotype: mean={G_df.values.mean():.6f}, std={G_df.values.std():.6f}")
        
        return G_df, snp_positions
    
    def _load_genotype_source(self) -> Tuple[pd.DataFrame, Any]:
        """Load raw genotype from source files."""
        from pandas_plink import read_plink
        
        paths = self.config.paths
        dset = self.config.dset
        
        if dset in ["thaliana_horton", "thaliana_1001"]:
            bim, self.fam, bed = read_plink(paths.geno)
            gen_data = pd.DataFrame(
                bed.compute().T,
                index=pd.MultiIndex.from_arrays(
                    [self.fam.fid.astype(int), self.fam.iid.astype(int)],
                    names=['fid', 'iid']
                )
            )
            return gen_data, bim
        
        # Custom dataset - try PLINK first
        from pathlib import Path
        geno_path = Path(paths.geno)
        
        if geno_path.suffix in ['.bed', '.bim', '.fam'] or \
           all(geno_path.with_suffix(s).exists() for s in ['.bed', '.bim', '.fam']):
            bim, self.fam, bed = read_plink(str(geno_path.with_suffix('')))
            gen_data = pd.DataFrame(
                bed.compute().T,
                index=pd.MultiIndex.from_arrays(
                    [self.fam.fid.astype(int), self.fam.iid.astype(int)],
                    names=['fid', 'iid']
                )
            )
            return gen_data, bim
        
        # CSV/TSV fallback
        sep = '\t' if geno_path.suffix in ['.tsv', '.txt'] else ','
        gen_data = pd.read_csv(paths.geno, sep=sep, index_col=0)
        gen_data = gen_data.fillna(gen_data.mean())
        return gen_data, None

    def _normalize_genotype(self, gen_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize genotype matrix (matching torch version). Returns DataFrame with index preserved."""
        from limix_modified.run_numpylimix._gaussianize_geno import normalize_genotype_matrix
        
        gen_data_normalized = normalize_genotype_matrix(gen_data)
        
        # Ensure numerical stability while keeping DataFrame
        values = gen_data_normalized.values
        
        # Return DataFrame with original index preserved
        return pd.DataFrame(values, index=gen_data.index, columns=gen_data_normalized.columns)
        
    def _subsample_for_simulation(
        self, 
        gen_data: pd.DataFrame, 
        snp_positions: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Subsample or stack individuals for simulation."""
        
        n_available = len(gen_data)
        n_target = self.config.sim_config.num_samples
        
        if n_available == n_target:
            return gen_data, snp_positions
        
        gen_data_sorted = gen_data.sort_index()
        
        if n_available > n_target:
            # Downsample
            indices = np.sort(self._rng.choice(n_available, size=n_target, replace=False))
            sampled_ids = gen_data_sorted.index[indices]
            self._log(f"[INFO] Subsampled {n_target} from {n_available} individuals")
            return gen_data_sorted.loc[sampled_ids], snp_positions
        
        # Upsample by stacking (n_target > n_available)
        full_copies = n_target // n_available
        remainder = n_target % n_available

        chunks = [gen_data_sorted] * full_copies
        if remainder > 0:
            extra_idx = np.sort(self._rng.choice(n_available, size=remainder, replace=False))
            chunks.append(gen_data_sorted.iloc[extra_idx])

        stacked = pd.concat(chunks, ignore_index=False)

        new_fids = np.arange(1, len(stacked) + 1)
        stacked.index = pd.MultiIndex.from_arrays(
            [new_fids, new_fids.copy()], names=['fid', 'iid']
        )

        self._log(
            f"[INFO] Stacked genotypes: {n_available} -> {len(stacked)} "
            f"({full_copies} full copies + {remainder} extra samples)"
        )
        return stacked, snp_positions
    
    def _load_snp_positions(self, bim: Any) -> Optional[pd.DataFrame]:
        """Load SNP position annotations."""
        # Simplified version - add full logic from torch version if needed
        if bim is not None and isinstance(bim, pd.DataFrame):
            if 'chrom' in bim.columns and 'pos' in bim.columns:
                return bim[['chrom', 'pos']]
        return None
    
    def _load_phenotype(self, G_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Load or simulate phenotype data. Returns DataFrame."""
        
        if self.config.sim_config.enabled:
            return self._generate_simulated_phenotype(G_df)
        else:
            return self._load_real_phenotype(), None

    def _generate_simulated_phenotype(self, G_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Generate simulated phenotype (matching torch version). Returns DataFrame."""
        
        sim_config = self.config.sim_config
        
        self._log(f"[INFO] Generating simulated phenotype: eta={sim_config.eta}, "
                f"rep_idx={sim_config.rep_idx}")
        
        from limix_modified.run_numpylimix.simulator import PhenoSimulator
        chrom, pos = self._extract_chrom_pos()
        
        simulator = PhenoSimulator(
            dset=self.config.dset,
            X=G_df,
            P=sim_config.num_tasks,
            eta=sim_config.eta,
            rep_idx=sim_config.rep_idx,
            chrom=chrom,
            pos=pos,
            reference_trait=sim_config.reference_trait
        )
        
        Xr, region, global_indices = simulator.getRegion(size=None)
        df, info = simulator.genPheno(
            Xr,
            ncausal=sim_config.ncausal,
            use_heterogeneity=sim_config.use_heterogeneity,
            corr_bounds=sim_config.corr_bounds,
            global_indices=global_indices
        )

        # Align phenotype index to (possibly stacked) genotype
        if len(df) == len(G_df):
            df.index = G_df.index

        return df, info

    def _load_real_phenotype(self) -> pd.DataFrame:
        """Load real phenotype data. Returns DataFrame with index."""
        paths = self.config.paths
        dset = self.config.dset
        
        if dset in ["thaliana_horton", "thaliana_1001"]:
            df = pd.read_csv(paths.pheno, sep="\t")
            df.set_index(['fid', 'iid'], inplace=True)
        else:
            # Custom dataset
            df = self._load_phenotype_custom()
        
        return df
    
    def _load_phenotype_custom(self) -> pd.DataFrame:
        """Load custom format phenotype."""
        paths = self.config.paths
        
        # Detect header
        with open(paths.pheno, 'r') as f:
            first_line = f.readline().strip().lower()
        
        cols = first_line.split('\t')
        has_header = len(cols) >= 2 and cols[0] == 'fid' and cols[1] == 'iid'
        
        df = pd.read_csv(paths.pheno, sep='\t', header=0 if has_header else None)
        
        # Standardize columns
        n_cols = df.shape[1]
        df.columns = ['fid', 'iid'] + [f'phenotype_{i}' for i in range(n_cols - 2)]
        
        df.set_index(['fid', 'iid'], inplace=True)
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0).astype(int),
             df.index.get_level_values(1).astype(int)],
            names=['fid', 'iid']
        )
        
        return df.fillna(df.mean())
    
    def _apply_transformation(self, Y: np.ndarray) -> np.ndarray:
        """Apply phenotype transformation (matching torch version)."""
        method = self.config.correction_config.transformation
        
        if method in [None, 'none']:
            return Y
        
        self._log(f"[INFO] Applying transformation: {method}")
        Y = standardize_data(Y, method=method)
        return Y
    
    def _align_samples(
        self,
        G_df: pd.DataFrame,
        Y_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
        """
        Align genotype and phenotype samples by index.
        Matches torch MultitaskDatasetSNP._align_sample_indices() exactly.
        
        Args:
            G_df: Genotype DataFrame with sample index
            Y_df: Phenotype DataFrame with sample index
            
        Returns:
            G_aligned: Aligned genotype DataFrame
            Y_aligned: Aligned phenotype DataFrame
            common_index: The common sample index used
        """
        
        if isinstance(G_df.index, pd.MultiIndex) and isinstance(Y_df.index, pd.MultiIndex):
            G_aligned, Y_aligned, common_index = self._align_multiindex(G_df, Y_df)
        else:
            G_aligned, Y_aligned, common_index = self._align_simple_index(G_df, Y_df)
        
        # Final verification
        if len(G_aligned) != len(Y_aligned):
            raise ValueError(
                f"Length mismatch after alignment: {len(G_aligned)} vs {len(Y_aligned)}"
            )
        
        return G_aligned, Y_aligned, common_index

    def _align_multiindex(
        self,
        G_df: pd.DataFrame,
        Y_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
        """
        Align MultiIndex samples deterministically.
        Uses numeric sorting for reproducibility across pandas versions.
        """
        # Convert to string tuples for reliable matching
        G_tuples = {(str(f), str(i)): (f, i) for f, i in G_df.index}
        Y_tuples = {(str(f), str(i)): (f, i) for f, i in Y_df.index}
        
        # Find common samples
        common_str = set(G_tuples.keys()) & set(Y_tuples.keys())
        
        if not common_str:
            raise ValueError("No common samples between genotype and phenotype!")
        
        common_sorted = sorted(
            [(int(f), int(i)) for f, i in common_str],
            key=lambda x: (x[0], x[1])
        )
        
        # Convert back to string for mapping lookup
        common_str_sorted = [(str(f), str(i)) for f, i in common_sorted]
        
        # Get original index tuples in sorted order
        G_idx = [G_tuples[t] for t in common_str_sorted]
        Y_idx = [Y_tuples[t] for t in common_str_sorted]
        
        # Subset both DataFrames
        G_aligned = G_df.loc[G_idx]
        Y_aligned = Y_df.loc[Y_idx]
        
        # Verify ordering matches
        G_final = [(str(f), str(i)) for f, i in G_aligned.index]
        Y_final = [(str(f), str(i)) for f, i in Y_aligned.index]
        
        if G_final != Y_final:
            raise ValueError("Sample ordering mismatch after alignment!")
        
        self._log(f"[INFO] Aligned {len(common_sorted)} MultiIndex samples")
        
        return G_aligned, Y_aligned, G_aligned.index

    def _align_simple_index(
        self,
        G_df: pd.DataFrame,
        Y_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
        """Align simple (non-MultiIndex) samples."""
        common = G_df.index.intersection(Y_df.index)
        
        if len(common) == 0:
            raise ValueError("No common samples between genotype and phenotype!")
        
        # Sort for determinism
        common = common.sort_values()
        
        G_aligned = G_df.loc[common]
        Y_aligned = Y_df.loc[common]
        
        self._log(f"[INFO] Aligned {len(common)} samples")
        
        return G_aligned, Y_aligned, common

    def _build_metadata(self, common_index: pd.Index) -> Dict[str, Any]:
        """Build metadata dictionary."""
        return {
            'simulation_info': self.simulation_info,
            'snp_positions': self.snp_positions,
            'seed': self.config.seed,
            'dset': self.config.dset,
            'sample_index': common_index,  # Store the aligned sample index
        }
    
    def _compute_kinship(self, G: np.ndarray) -> np.ndarray:
        """Compute kinship matrix K = GG'/p."""
        n, p = G.shape
        K = G @ G.T / p
        
        trace = np.trace(K)
        self._log(f"[INFO] Kinship: trace={trace:.4f}, expected={n}")
        
        return K
    
    def _extract_chrom_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract chromosome and position arrays."""
        if self.snp_positions is None:
            return np.array([]), np.array([])
        
        df = self.snp_positions
        if 'chrom' in df.columns and 'pos' in df.columns:
            return df['chrom'].values, df['pos'].values
        
        return np.array([]), np.array([])


def get_data_numpy(
    config: Dict[str, Any],
    rep_idx: Optional[int] = None,
    eta: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load data for LIMIX numpy analysis.
    
    Drop-in replacement matching torch dataloader behavior.
    
    Args:
        config: Configuration dictionary
        rep_idx: Repetition index for simulation
        eta: Eta value for simulation
    
    Returns:
        G: Genotype matrix
        K: Kinship matrix
        Y: Phenotype matrix
        meta: Metadata dictionary
    """
    data_param = config.get('data_param', {})
    
    # Build paths
    paths = DataPaths(
        pheno=data_param.get('pheno_path'),
        geno=data_param.get('geno_path'),
        annot=data_param.get('annot_path'),
        batch=data_param.get('batch_path'),
        cov=data_param.get('cov_path'),
        config=data_param.get('data_path_config'),
    )
    
    # Resolve paths from config if needed
    if paths.config and (paths.pheno is None or paths.geno is None):
        paths = _resolve_paths_from_config(paths, data_param.get('dset', 'custom'))
    
    # Build simulation config
    sim_config = SimulationConfig(
        enabled=data_param.get('simulated', False),
        num_samples=data_param.get('num_samples', 250),
        eta=eta if eta is not None else data_param.get('eta', 0.5),
        ncausal=data_param.get('ncausal', 1),
        corr_bounds=data_param.get('corr_bounds', 0.0),
        use_heterogeneity=data_param.get('use_heterogeneity', False),
        num_tasks=data_param.get('num_tasks', 2),
        reference_trait=data_param.get('reference_trait'),
        rep_idx=rep_idx
    )
    
    # Build correction config
    correction_config = CorrectionConfig(
        regress_batch=data_param.get('regress_out_batch_effects', True),
        regress_covariates=data_param.get('regress_out_covariates', True),
        per_trait_batch=data_param.get('per_trait_batch', False),
        transformation=data_param.get('transformation_method', 'none'),
    )
    
    # Build full config
    data_config = NumpyDataConfig(
        dset=data_param.get('dset', 'custom'),
        seed=data_param.get('seed', 42),
        paths=paths,
        sim_config=sim_config,
        correction_config=correction_config,
        root=data_param.get('root'),
    )
    
    # Load data
    loader = NumpyDataLoader(data_config, verbose=config.get('verbose', True))
    G, K, Y, meta = loader.load()
    
    return G, K, Y, meta


# Helper functions
def get_package_root() -> Path:
    """Get the root directory of the run_numpylimix package."""
    return Path(resources.files("limix_modified"))


def resolve_path_placeholders(path: Optional[str]) -> Optional[str]:
    """Replace ${PACKAGE_ROOT} placeholder with actual package root path."""
    if path is None:
        return None
    package_root = str(get_package_root())
    resolved = path.replace("${PACKAGE_ROOT}", package_root)
    resolved = os.path.expanduser(resolved)
    return resolved

def _resolve_paths_from_config(paths: DataPaths, dset: str) -> DataPaths:
    """Resolve paths from JSON config file."""
    import json
    
    # Resolve the config path itself first
    config_path = resolve_path_placeholders(paths.config)
    
    # Load and resolve all paths in config
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Replace placeholder in the entire config file
    config_text = config_text.replace("${PACKAGE_ROOT}", str(get_package_root()))
    config = json.loads(config_text)
    
    path_mappings = {
        "thaliana_horton": "thaliana_horton_geno_path",
        "thaliana_1001": "thaliana_1001_geno_path",
    }
    
    if dset in path_mappings:
        geno_key = path_mappings[dset]
        paths.geno = config[geno_key]
        
        if dset == "thaliana_horton" and not paths.batch:
            paths.batch = config.get("thaliana_horton_batch_path")
    else:
        paths.geno = config.get(f"{dset}_geno_path")
        paths.batch = config.get(f"{dset}_batch_path")
    
    return paths