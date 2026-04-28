import numpy as np
from typing import Tuple
from numpy import finfo, sqrt

def prepare_kinship_pipeline(
    G: np.ndarray,
    epsilon: float = sqrt(finfo(float).eps),
    debug: bool = False,
    chunk_size: int = 100000,  
) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], np.ndarray]:
    """
    Prepare kinship matrix with memory-efficient chunked computation using NumPy.
    
    Parameters
    ----------
    G : np.ndarray
        Raw genotype matrix (N individuals × S SNPs)
    epsilon : float
        Small value for numerical stability
    debug : bool
        Print debug information
    chunk_size : int
        Number of SNPs to process at once (reduce if memory issues persist)
        
    Returns
    -------
    K : np.ndarray
        Kinship matrix (N × N)
    QS : Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
        ((Q0, Q1), S0) - Eigendecomposition components
    G_lowrank : np.ndarray
        Low-rank representation
    """
    
    G = G.astype(np.float64)
    N, S = G.shape
    
    if S > chunk_size and debug:
        print(f"[DEBUG] Computing kinship in chunks of {chunk_size} SNPs...")
    
    # Initialize kinship matrix
    K_full = np.zeros((N, N), dtype=G.dtype)
    
    # Compute K = G @ G.T in chunks to save memory
    num_chunks = (S + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, S)
        
        G_chunk = G[:, start_idx:end_idx]
        K_full += G_chunk @ G_chunk.T
        
        if debug and (num_chunks > 1):
            if (i + 1) % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                print(f"[DEBUG] Kinship progress: {end_idx}/{S} SNPs ({100*end_idx/S:.1f}%)")
        
        del G_chunk
    
    K_full = K_full / S
    
    if debug:
        print(f"[DEBUG] K_full shape: {K_full.shape}")
        print(f"[DEBUG] K_full diagonal mean: {np.diag(K_full).mean():.4f}")
    
    # Eigendecomposition with scipy fallback (matching economic_qs)
    S_full, Q_full = np.linalg.eigh(K_full)
    
    # Check if numpy result looks suspicious (matching economic_qs logic)
    nok = abs(max(Q_full[0].min(), Q_full[0].max(), key=abs)) < epsilon
    nok = nok and abs(max(K_full.min(), K_full.max(), key=abs)) >= epsilon
    
    if nok:
        if debug:
            print("[DEBUG] NumPy eigh result suspicious, falling back to SciPy")
        from scipy.linalg import eigh as sp_eigh
        S_full, Q_full = sp_eigh(K_full)
    
    if debug:
        print(f"[DEBUG] Original eigenvalues: {len(S_full)} total")
        print(f"[DEBUG] Eigenvalue range: [{S_full.min():.6e}, {S_full.max():.6f}]")
        print(f"[DEBUG] Eigenvalues below threshold ({epsilon:.2e}): {(S_full < epsilon).sum()}")
    
    # Filter eigenvalues (matching economic_qs)
    ok = S_full >= epsilon
    nok = ~ok
    
    S0 = S_full[ok]
    Q0 = Q_full[:, ok]
    Q1 = Q_full[:, nok] if nok.sum() > 0 else np.empty((Q_full.shape[0], 0), dtype=Q_full.dtype)
    
    if debug:
        print(f"[DEBUG] Filtered {(~ok).sum()} eigenvalues below {epsilon:.2e}")
        print(f"[DEBUG] Kept {ok.sum()} eigenvalues")
    
    # Reconstruct PSD-corrected K
    G_lowrank = Q0 @ np.diag(np.sqrt(S0))
    K = G_lowrank @ G_lowrank.T
    
    return K, ((Q0, Q1), S0), G_lowrank