import warnings

from limix_modified._data._conform import conform_dataset
from limix_modified._data._lik import normalize_likelihood
from limix_modified._display._session import session_block
import numpy as np
import pandas as pd

from functools import reduce

from numpy import asarray, asfortranarray, kron, log, sqrt, tensordot, trace
from numpy.linalg import inv, matrix_rank, slogdet
from optimix import Function
from glimix_core._util import cached_property, log2pi, unvec, vec
from glimix_core.cov import Kron2SumCov
from glimix_core.mean import KronMean

from glimix_core._util import lu_factor, lu_solve
from glimix_core.lmm._kron2sum_scan import KronFastScanner


class Kron2Sum(Function):
    """
    LMM for multi-traits fitted via maximum likelihood.

    This implementation follows the work published in [CA05]_.
    Let n, c, and p be the number of samples, covariates, and traits, respectively.
    The outcome variable Y is a n×p matrix distributed according to::

        vec(Y) ~ N((A ⊗ X) vec(B), K = C₀ ⊗ GGᵀ + C₁ ⊗ I).

    A and X are design matrices of dimensions p×p and n×c provided by the user,
    where X is the usual matrix of covariates commonly used in single-trait models.
    B is a c×p matrix of fixed-effect sizes per trait.
    G is a n×r matrix provided by the user and I is a n×n identity matrices.
    C₀ and C₁ are both symmetric matrices of dimensions p×p, for which C₁ is
    guaranteed by our implementation to be of full rank.
    The parameters of this model are the matrices B, C₀, and C₁.

    For implementation purpose, we make use of the following definitions:

    - 𝛃 = vec(B)
    - M = A ⊗ X
    - H = MᵀK⁻¹M
    - Yₓ = LₓY
    - Yₕ = YₓLₕᵀ
    - Mₓ = LₓX
    - Mₕ = (LₕA) ⊗ Mₓ
    - mₕ = Mₕvec(B)

    where Lₓ and Lₕ are defined in :class:`glimix_core.cov.Kron2SumCov`.

    References
    ----------
    .. [CA05] Casale, F. P., Rakitsch, B., Lippert, C., & Stegle, O. (2015). Efficient
       set tests for the genetic analysis of correlated traits. Nature methods, 12(8),
       755.
    """

    def __init__(self, Y, A, X, G, rank=1, restricted=False):
        """
        Constructor.

        Parameters
        ----------
        Y : (n, p) array_like
            Outcome matrix.
        A : (p, p) array_like
            Trait-by-trait design matrix.
        X : (n, c) array_like
            Covariates design matrix.
        G : (n, r) array_like
            Matrix G from the GGᵀ term.
        rank : optional, int
            Maximum rank of matrix C₀. Defaults to ``1``.
        """
        from numpy_sugar import is_all_finite

        Y = asfortranarray(Y, float)
        yrank = matrix_rank(Y)
        if Y.shape[1] > yrank:
            warnings.warn(
                f"Y is not full column rank: rank(Y)={yrank}. "
                + "Convergence might be problematic.",
                UserWarning,
            )

        A = asarray(A, float)
        X = asarray(X, float)
        Xrank = matrix_rank(X)
        if X.shape[1] > Xrank:
            warnings.warn(
                f"X is not full column rank: rank(X)={Xrank}. "
                + "Convergence might be problematic.",
                UserWarning,
            )
        G = asarray(G, float).copy()
        self._G_norm = max(G.min(), G.max())
        G /= self._G_norm

        if not is_all_finite(Y):
            raise ValueError("There are non-finite values in the outcome matrix.")

        if not is_all_finite(A):
            msg = "There are non-finite values in the trait-by-trait design matrix."
            raise ValueError(msg)

        if not is_all_finite(X):
            raise ValueError("There are non-finite values in the covariates matrix.")

        if not is_all_finite(G):
            raise ValueError("There are non-finite values in the G matrix.")

        self._Y = Y
        self._cov = Kron2SumCov(G, Y.shape[1], rank)
        self._cov.listen(self._parameters_update)
        self._mean = KronMean(A, X)
        self._cache = {"terms": None}
        self._restricted = restricted
        composite = [("C0", self._cov.C0), ("C1", self._cov.C1)]
        Function.__init__(self, "Kron2Sum", composite=composite)

        nparams = self._mean.nparams + self._cov.nparams
        if nparams > Y.size:
            msg = "The number of parameters is larger than the outcome size."
            msg += " Convergence is expected to be problematic."
            warnings.warn(msg, UserWarning)

    @property
    def beta_covariance(self):
        """
        Estimates the covariance-matrix of the optimal beta.

        Returns
        -------
        beta-covariance : ndarray
            (MᵀK⁻¹M)⁻¹.

        References
        ----------
        .. Rencher, A. C., & Schaalje, G. B. (2008). Linear models in statistics. John
           Wiley & Sons.
        """
        H = self._terms["H"]
        return inv(H)

    def get_fast_scanner(self):
        """
        Return :class:`.FastScanner` for association scan.

        Returns
        -------
        :class:`.FastScanner`
            Instance of a class designed to perform very fast association scan.
        """
        terms = self._terms
        return KronFastScanner(self._Y, self._mean.A, self._mean.X, self._cov.Ge, terms)

    @property
    def A(self):
        """
        A from the equation 𝐦 = (A ⊗ X) vec(B).

        Returns
        -------
        A : ndarray
            A.
        """
        return self._mean.A

    @property
    def B(self):
        """
        Fixed-effect sizes B from 𝐦 = (A ⊗ X) vec(B).

        Returns
        -------
        fixed-effects : ndarray
            B from 𝐦 = (A ⊗ X) vec(B).
        """
        self._terms
        return asarray(self._mean.B, float)

    @property
    def beta(self):
        """
        Fixed-effect sizes 𝛃 = vec(B).

        Returns
        -------
        fixed-effects : ndarray
            𝛃 from 𝛃 = vec(B).
        """
        return vec(self.B)

    @property
    def C0(self):
        """
        C₀ from equation K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

        Returns
        -------
        C0 : ndarray
            C₀.
        """
        return self._cov.C0.value() / (self._G_norm**2)

    @property
    def C1(self):
        """
        C₁ from equation K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

        Returns
        -------
        C1 : ndarray
            C₁.
        """
        return self._cov.C1.value()

    def mean(self):
        """
        Mean 𝐦 = (A ⊗ X) vec(B).

        Returns
        -------
        mean : ndarray
            𝐦.
        """
        self._terms
        return self._mean.value()

    def covariance(self):
        """
        Covariance K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

        Returns
        -------
        covariance : ndarray
            K.
        """
        return self._cov.value()

    @property
    def X(self):
        """
        X from equation M = (A ⊗ X).

        Returns
        -------
        X : ndarray
            X from M = (A ⊗ X).
        """
        return self._mean.X

    @property
    def M(self):
        """
        M = (A ⊗ X).

        Returns
        -------
        M : ndarray
            M from M = (A ⊗ X).
        """
        return self._mean.AX

    @property
    def nsamples(self):
        """
        Number of samples, n.
        """
        return self._Y.shape[0]

    @property
    def ntraits(self):
        """
        Number of traits, p.
        """
        return self._Y.shape[1]

    @property
    def ncovariates(self):
        """
        Number of covariates, c.
        """
        return self._mean.X.shape[1]

    def value(self):
        """
        Log of the marginal likelihood.
        """
        return self.lml()

    def gradient(self):
        """
        Gradient of the log of the marginal likelihood.
        """
        return self._lml_gradient()

    def lml(self):
        """
        Log of the marginal likelihood.
        Let 𝐲 = vec(Y), M = A⊗X, and H = MᵀK⁻¹M. The restricted log of the marginal
        likelihood is given by [R07]_::
        2⋅log(p(𝐲)) = -(n⋅p - c⋅p) log(2π) + log(｜MᵀM｜) - log(｜K｜) - log(｜H｜)
        - (𝐲-𝐦)ᵀ K⁻¹ (𝐲-𝐦),
        where 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵀK⁻¹𝐲.
        For implementation purpose, let X = (L₀ ⊗ G) and R = (L₁ ⊗ I)(L₁ ⊗ I)ᵀ.
        The covariance can be written as::
        K = XXᵀ + R.
        From the Woodbury matrix identity, we have
        𝐲ᵀK⁻¹𝐲 = 𝐲ᵀR⁻¹𝐲 - 𝐲ᵀR⁻¹XZ⁻¹XᵀR⁻¹𝐲,
        where Z = I + XᵀR⁻¹X. Note that R⁻¹ = (U₁S₁⁻¹U₁ᵀ) ⊗ I and ::
        XᵀR⁻¹𝐲 = (L₀ᵀW ⊗ Gᵀ)𝐲 = vec(GᵀYWL₀),
        where W = U₁S₁⁻¹U₁ᵀ. The term GᵀY can be calculated only once and it will form a
        r×p matrix. We similarly have ::
        XᵀR⁻¹M = (L₀ᵀWA) ⊗ (GᵀX),
        for which GᵀX is pre-computed.
        The log-determinant of the covariance matrix is given by
        log(｜K｜) = log(｜Z｜) - log(｜R⁻¹｜) = log(｜Z｜) - 2·n·log(｜U₁S₁⁻½｜).
        The log of the marginal likelihood can be rewritten as::
        2⋅log(p(𝐲)) = -(n⋅p - c⋅p) log(2π) + log(｜MᵀM｜)
        - log(｜Z｜) + 2·n·log(｜U₁S₁⁻½｜)
        - log(｜MᵀR⁻¹M - MᵀR⁻¹XZ⁻¹XᵀR⁻¹M｜)
        - 𝐲ᵀR⁻¹𝐲 + (𝐲ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐲)
        - 𝐦ᵀR⁻¹𝐦 + (𝐦ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐦)
        + 2𝐲ᵀR⁻¹𝐦 - 2(𝐲ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐦).
        Returns
        -------
        lml : float
            Log of the marginal likelihood.
        References
        ----------
        .. [R07] LaMotte, L. R. (2007). A direct derivation of the REML likelihood
                 function. Statistical Papers, 48(2), 321-327.
        """
        print("=== Starting Log Marginal Likelihood Computation ===")
        
        # Get terms dictionary
        terms = self._terms
    
        # Extract quadratic form terms
        yKiy = terms["yKiy"]
        print(f"DEBUG: yKiy (y^T K^(-1) y) = {yKiy}")
        print(f"DEBUG: yKiy type: {type(yKiy)}")
        
        mKiy = terms["mKiy"]
        print(f"DEBUG: mKiy (m^T K^(-1) y) = {mKiy}")
        print(f"DEBUG: mKiy type: {type(mKiy)}")
        
        mKim = terms["mKim"]
        print(f"DEBUG: mKim (m^T K^(-1) m) = {mKim}")
        print(f"DEBUG: mKim type: {type(mKim)}")
        
        # Get degrees of freedom
        df = self._df
        print(f"DEBUG: _df (degrees of freedom) = {df}")
        print(f"DEBUG: _df type: {type(df)}")
        
        # Get log determinant terms
        logdet_MM = self._logdet_MM
        print(f"DEBUG: _logdet_MM (log|M^T M|) = {logdet_MM}")
        print(f"DEBUG: _logdet_MM type: {type(logdet_MM)}")
        
        logdetK = self._logdetK
        print(f"DEBUG: _logdetK (log|K|) = {logdetK}")
        print(f"DEBUG: _logdetK type: {type(logdetK)}")
        
        logdetH = self._logdetH
        print(f"DEBUG: _logdetH (log|H|) = {logdetH}")
        print(f"DEBUG: _logdetH type: {type(logdetH)}")
        
        # Calculate log(2π) term
        log2pi_term = -df * log2pi
        print(f"DEBUG: log2pi constant = {log2pi}")
        print(f"DEBUG: -df * log2pi = {log2pi_term}")
        
        # Start building the log marginal likelihood
        print("\n=== Computing LML Components ===")
        
        # Initialize with degrees of freedom term
        lml = log2pi_term
        print(f"DEBUG: lml after df term = {lml}")
        
        # Add log determinant of M^T M
        lml += logdet_MM
        print(f"DEBUG: lml after adding logdet_MM = {lml}")
        
        # Subtract log determinant of K
        lml -= logdetK
        print(f"DEBUG: lml after subtracting logdetK = {lml}")
        
        # Subtract log determinant of H
        lml -= logdetH
        print(f"DEBUG: lml after subtracting logdetH = {lml}")
        
        # Subtract quadratic forms
        print(f"DEBUG: About to subtract yKiy: {yKiy}")
        lml += -yKiy
        print(f"DEBUG: lml after subtracting yKiy = {lml}")
        
        print(f"DEBUG: About to subtract mKim: {mKim}")
        lml -= mKim
        print(f"DEBUG: lml after subtracting mKim = {lml}")
        
        # Add cross term
        cross_term = 2 * mKiy
        print(f"DEBUG: cross term (2 * mKiy) = {cross_term}")
        lml += cross_term
        print(f"DEBUG: lml after adding cross term = {lml}")
        
        print(f"\nDEBUG: Final lml before division by 2 = {lml}")
        
        # Divide by 2 for final result
        final_lml = lml / 2
        print(f"DEBUG: Final lml after division by 2 = {final_lml}")
        print(f"DEBUG: Final lml type: {type(final_lml)}")
        
        print("=== LML Computation Complete ===\n")
        
        return final_lml
    

    def fit(self, verbose=True):
        """
        Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool, optional
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        self._maximize(verbose=verbose, factr=1e7, pgtol=1e-7)

    def _parameters_update(self):
        self._cache["terms"] = None

    @cached_property
    def _GY(self):
        return self._cov.Ge.T @ self._Y

    @cached_property
    def _GG(self):
        return self._cov.Ge.T @ self._cov.Ge

    @cached_property
    def _trGG(self):
        from numpy_sugar.linalg import trace2

        return trace2(self._cov.Ge, self._cov.Ge.T)

    @cached_property
    def _GGGG(self):
        return self._GG @ self._GG

    @cached_property
    def _GGGY(self):
        return self._GG @ self._GY

    @cached_property
    def _XX(self):
        return self._mean.X.T @ self._mean.X

    @cached_property
    def _GX(self):
        return self._cov.Ge.T @ self._mean.X

    @cached_property
    def _XGGG(self):
        return self._GX.T @ self._GG

    @cached_property
    def _XGGY(self):
        return self._GX.T @ self._GY

    @cached_property
    def _XGGX(self):
        return self._GX.T @ self._GX

    @cached_property
    def _XY(self):
        return self._mean.X.T @ self._Y

    @property
    def _terms(self):
        from numpy_sugar.linalg import ddot, lu_slogdet, sum2diag

        if self._cache["terms"] is not None:
            return self._cache["terms"]

        L0 = self._cov.C0.L
        S, U = self._cov.C1.eigh()
        W = ddot(U, 1 / S) @ U.T
        S = 1 / sqrt(S)
        Y = self._Y
        A = self._mean.A

        WL0 = W @ L0
        YW = Y @ W
        WA = W @ A
        L0WA = L0.T @ WA

        Z = kron(L0.T @ WL0, self._GG)
        Z = sum2diag(Z, 1)
        Lz = lu_factor(Z, check_finite=False)

        # 𝐲ᵀR⁻¹𝐲 = vec(YW)ᵀ𝐲
        yRiy = (YW * self._Y).sum()
        # MᵀR⁻¹M = AᵀWA ⊗ XᵀX
        MRiM = kron(A.T @ WA, self._XX)
        # XᵀR⁻¹𝐲 = vec(GᵀYWL₀)
        XRiy = vec(self._GY @ WL0)
        # XᵀR⁻¹M = (L₀ᵀWA) ⊗ (GᵀX)
        XRiM = kron(L0WA, self._GX)
        # MᵀR⁻¹𝐲 = vec(XᵀYWA)
        MRiy = vec(self._XY @ WA)

        ZiXRiM = lu_solve(Lz, XRiM)
        ZiXRiy = lu_solve(Lz, XRiy)

        MRiXZiXRiy = ZiXRiM.T @ XRiy
        MRiXZiXRiM = XRiM.T @ ZiXRiM

        yKiy = yRiy - XRiy @ ZiXRiy
        MKiy = MRiy - MRiXZiXRiy
        H = MRiM - MRiXZiXRiM
        Lh = lu_factor(H, check_finite=False)
        b = lu_solve(Lh, MKiy)
        B = unvec(b, (self.ncovariates, -1))
        self._mean.B = B
        XRim = XRiM @ b

        ZiXRim = ZiXRiM @ b
        mRiy = b.T @ MRiy
        mRim = b.T @ MRiM @ b

        logdetK = lu_slogdet(Lz)[1]
        logdetK -= 2 * log(S).sum() * self.nsamples

        mKiy = mRiy - XRim.T @ ZiXRiy
        mKim = mRim - XRim.T @ ZiXRim

        self._cache["terms"] = {
            "logdetK": logdetK,
            "mKiy": mKiy,
            "mKim": mKim,
            "b": b,
            "Z": Z,
            "B": B,
            "Lz": Lz,
            "S": S,
            "W": W,
            "WA": WA,
            "YW": YW,
            "WL0": WL0,
            "yRiy": yRiy,
            "MRiM": MRiM,
            "XRiy": XRiy,
            "XRiM": XRiM,
            "ZiXRiM": ZiXRiM,
            "ZiXRiy": ZiXRiy,
            "ZiXRim": ZiXRim,
            "MRiy": MRiy,
            "mRim": mRim,
            "mRiy": mRiy,
            "XRim": XRim,
            "yKiy": yKiy,
            "H": H,
            "Lh": Lh,
            "MRiXZiXRiy": MRiXZiXRiy,
            "MRiXZiXRiM": MRiXZiXRiM,
        }
        return self._cache["terms"]
    
    def _lml_gradient(self):
            """
            Gradient of the log of the marginal likelihood.

            Let 𝐲 = vec(Y), 𝕂 = K⁻¹∂(K)K⁻¹, and H = MᵀK⁻¹M. The gradient is given by::

                2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) - tr(H⁻¹∂H) + 𝐲ᵀ𝕂𝐲 - 𝐦ᵀ𝕂(2⋅𝐲-𝐦)
                    - 2⋅(𝐦-𝐲)ᵀK⁻¹∂(𝐦).

            Observe that

                ∂𝛃 = -H⁻¹(∂H)𝛃 - H⁻¹Mᵀ𝕂𝐲 and ∂H = -Mᵀ𝕂M.

            Let Z = I + XᵀR⁻¹X and 𝓡 = R⁻¹(∂K)R⁻¹. We use Woodbury matrix identity to
            write ::

                𝐲ᵀ𝕂𝐲 = 𝐲ᵀ𝓡𝐲 - 2(𝐲ᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲) + (𝐲ᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
                Mᵀ𝕂M = Mᵀ𝓡M - 2(Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹M) + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹M)
                Mᵀ𝕂𝐲 = Mᵀ𝓡𝐲 - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡𝐲) - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
                      + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
                H⁻¹   = MᵀR⁻¹M - (MᵀR⁻¹X)Z⁻¹(XᵀR⁻¹M),

            where we have used parentheses to separate expressions
            that we will compute separately. For example, we have ::

                𝐲ᵀ𝓡𝐲 = 𝐲ᵀ(U₁S₁⁻¹U₁ᵀ ⊗ I)(∂C₀ ⊗ GGᵀ)(U₁S₁⁻¹U₁ᵀ ⊗ I)𝐲
                      = 𝐲ᵀ(U₁S₁⁻¹U₁ᵀ∂C₀ ⊗ G)(U₁S₁⁻¹U₁ᵀ ⊗ Gᵀ)𝐲
                      = vec(GᵀYU₁S₁⁻¹U₁ᵀ∂C₀)ᵀvec(GᵀYU₁S₁⁻¹U₁ᵀ),

            when the derivative is over the parameters of C₀. Otherwise, we have

                𝐲ᵀ𝓡𝐲 = vec(YU₁S₁⁻¹U₁ᵀ∂C₁)ᵀvec(YU₁S₁⁻¹U₁ᵀ).

            The above equations can be more compactly written as

                𝐲ᵀ𝓡𝐲 = vec(EᵢᵀYW∂Cᵢ)ᵀvec(EᵢᵀYW),

            where W = U₁S₁⁻¹U₁ᵀ, E₀ = G, and E₁ = I. We will now just state the results for
            the other instances of the aBc form, which follow similar derivations::

                Xᵀ𝓡X = (L₀ᵀW∂CᵢWL₀) ⊗ (GᵀEᵢEᵢᵀG)
                Mᵀ𝓡y = (AᵀW∂Cᵢ⊗XᵀEᵢ)vec(EᵢᵀYW) = vec(XᵀEᵢEᵢᵀYW∂CᵢWA)
                Mᵀ𝓡X = AᵀW∂CᵢWL₀ ⊗ XᵀEᵢEᵢᵀG
                Mᵀ𝓡M = AᵀW∂CᵢWA ⊗ XᵀEᵢEᵢᵀX
                Xᵀ𝓡𝐲 = GᵀEᵢEᵢᵀYW∂CᵢWL₀

            From Woodbury matrix identity and Kronecker product properties we have ::

                tr(K⁻¹∂K) = tr[W∂Cᵢ]tr[EᵢEᵢᵀ] - tr[Z⁻¹(Xᵀ𝓡X)]
                tr(H⁻¹∂H) = - tr[(MᵀR⁻¹M)(Mᵀ𝕂M)] + tr[(MᵀR⁻¹X)Z⁻¹(XᵀR⁻¹M)(Mᵀ𝕂M)]

            Note also that ::

                ∂𝛃 = H⁻¹Mᵀ𝕂M𝛃 - H⁻¹Mᵀ𝕂𝐲.

            Returns
            -------
            C0.Lu : ndarray
                Gradient of the log of the marginal likelihood over C₀ parameters.
            C1.Lu : ndarray
                Gradient of the log of the marginal likelihood over C₁ parameters.
            """
            from numpy_sugar.linalg import lu_solve

            print("=== LML GRADIENT COMPUTATION START ===")

            terms = self._terms
            dC0 = self._cov.C0.gradient()["Lu"]
            dC1 = self._cov.C1.gradient()["Lu"]

            print(f"Initial gradients shapes: dC0={dC0.shape}, dC1={dC1.shape}")
            print(f"dC0 stats: min={dC0.min():.6f}, max={dC0.max():.6f}, mean={dC0.mean():.6f}")
            print(f"dC1 stats: min={dC1.min():.6f}, max={dC1.max():.6f}, mean={dC1.mean():.6f}")

            b = terms["b"]
            W = terms["W"]
            Lh = terms["Lh"]
            Lz = terms["Lz"]
            WA = terms["WA"]
            WL0 = terms["WL0"]
            YW = terms["YW"]
            MRiM = terms["MRiM"]
            MRiy = terms["MRiy"]
            XRiM = terms["XRiM"]
            XRiy = terms["XRiy"]
            ZiXRiM = terms["ZiXRiM"]
            ZiXRiy = terms["ZiXRiy"]

            print(f"\nPre-computed terms shapes:")
            print(f"  b: {b.shape}, W: {W.shape}")
            print(f"  WA: {WA.shape}, WL0: {WL0.shape}, YW: {YW.shape}")
            print(f"  MRiM: {MRiM.shape}, MRiy: {MRiy.shape}")
            print(f"  XRiM: {XRiM.shape}, XRiy: {XRiy.shape}")
            print(f"  ZiXRiM: {ZiXRiM.shape}, ZiXRiy: {ZiXRiy.shape}")

            # W derivatives computation
            WdC0 = _mdot(W, dC0)
            WdC1 = _mdot(W, dC1)

            print(f"\n--- W Derivatives ---")
            print(f"WdC0 shape: {WdC0.shape}, stats: min={WdC0.min():.6f}, max={WdC0.max():.6f}")
            print(f"WdC1 shape: {WdC1.shape}, stats: min={WdC1.min():.6f}, max={WdC1.max():.6f}")

            AWdC0 = _mdot(WA.T, dC0)
            AWdC1 = _mdot(WA.T, dC1)

            print(f"AWdC0 shape: {AWdC0.shape}, stats: min={AWdC0.min():.6f}, max={AWdC0.max():.6f}")
            print(f"AWdC1 shape: {AWdC1.shape}, stats: min={AWdC1.min():.6f}, max={AWdC1.max():.6f}")

            # Mᵀ𝓡M computation
            MR0M = _mkron(_mdot(AWdC0, WA), self._XGGX)
            MR1M = _mkron(_mdot(AWdC1, WA), self._XX)

            print(f"\n--- Mᵀ𝓡M Terms ---")
            print(f"MR0M shape: {MR0M.shape}, stats: min={MR0M.min():.6f}, max={MR0M.max():.6f}")
            print(f"MR1M shape: {MR1M.shape}, stats: min={MR1M.min():.6f}, max={MR1M.max():.6f}")
            print(f"_mdot(AWdC0, WA) shape: {_mdot(AWdC0, WA).shape}")
            print(f"_mdot(AWdC1, WA) shape: {_mdot(AWdC1, WA).shape}")
            print(f"self._XGGX shape: {self._XGGX.shape}, self._XX shape: {self._XX.shape}")

            # Mᵀ𝓡X computation
            MR0X = _mkron(_mdot(AWdC0, WL0), self._XGGG)
            MR1X = _mkron(_mdot(AWdC1, WL0), self._GX.T)

            print(f"\n--- Mᵀ𝓡X Terms ---")
            print(f"MR0X shape: {MR0X.shape}, stats: min={MR0X.min():.6f}, max={MR0X.max():.6f}")
            print(f"MR1X shape: {MR1X.shape}, stats: min={MR1X.min():.6f}, max={MR1X.max():.6f}")

            # Mᵀ𝓡𝐲 = (AᵀW∂Cᵢ⊗XᵀEᵢ)vec(EᵢᵀYW) = vec(XᵀEᵢEᵢᵀYW∂CᵢWA)
            MR0y = vec(_mdot(self._XGGY, _mdot(WdC0, WA)))
            MR1y = vec(_mdot(self._XY, WdC1, WA))

            print(f"\n--- Mᵀ𝓡𝐲 Terms ---")
            print(f"MR0y shape: {MR0y.shape}, stats: min={MR0y.min():.6f}, max={MR0y.max():.6f}")
            print(f"MR1y shape: {MR1y.shape}, stats: min={MR1y.min():.6f}, max={MR1y.max():.6f}")
            print(f"_mdot(self._XGGY, _mdot(WdC0, WA)) intermediate shape: {_mdot(self._XGGY, _mdot(WdC0, WA)).shape}")
            print(f"_mdot(self._XY, WdC1, WA) intermediate shape: {_mdot(self._XY, WdC1, WA).shape}")

            # Xᵀ𝓡X computation
            XR0X = _mkron(_mdot(WL0.T, dC0, WL0), self._GGGG)
            XR1X = _mkron(_mdot(WL0.T, dC1, WL0), self._GG)

            print(f"\n--- Xᵀ𝓡X Terms ---")
            print(f"XR0X shape: {XR0X.shape}, stats: min={XR0X.min():.6f}, max={XR0X.max():.6f}")
            print(f"XR1X shape: {XR1X.shape}, stats: min={XR1X.min():.6f}, max={XR1X.max():.6f}")
            print(f"_mdot(WL0.T, dC0, WL0) shape: {_mdot(WL0.T, dC0, WL0).shape}")
            print(f"_mdot(WL0.T, dC1, WL0) shape: {_mdot(WL0.T, dC1, WL0).shape}")

            # Xᵀ𝓡𝐲 computation
            XR0y = vec(_mdot(self._GGGY, WdC0, WL0))
            XR1y = vec(_mdot(self._GY, WdC1, WL0))

            print(f"\n--- Xᵀ𝓡𝐲 Terms ---")
            print(f"XR0y shape: {XR0y.shape}, stats: min={XR0y.min():.6f}, max={XR0y.max():.6f}")
            print(f"XR1y shape: {XR1y.shape}, stats: min={XR1y.min():.6f}, max={XR1y.max():.6f}")

            # 𝐲ᵀ𝓡𝐲 = vec(EᵢᵀYW∂Cᵢ)ᵀvec(EᵢᵀYW) computation
            yR0y = vec(_mdot(self._GY, WdC0)).T @ vec(self._GY @ W)
            yR1y = (YW.T * _mdot(self._Y, WdC1).T).T.sum(axis=(0, 1))

            print(f"\n--- 𝐲ᵀ𝓡𝐲 Terms ---")
            print(f"yR0y shape: {yR0y.shape}, value: {yR0y}")
            print(f"yR1y shape: {yR1y.shape}, value: {yR1y}")
            print(f"vec(_mdot(self._GY, WdC0)) shape: {vec(_mdot(self._GY, WdC0)).shape}")
            print(f"vec(self._GY @ W) shape: {vec(self._GY @ W).shape}")

            # Z inverse operations
            ZiXR0X = lu_solve(Lz, XR0X)
            ZiXR1X = lu_solve(Lz, XR1X)
            ZiXR0y = lu_solve(Lz, XR0y)
            ZiXR1y = lu_solve(Lz, XR1y)

            print(f"\n--- Z⁻¹ Operations ---")
            print(f"ZiXR0X shape: {ZiXR0X.shape}, stats: min={ZiXR0X.min():.6f}, max={ZiXR0X.max():.6f}")
            print(f"ZiXR1X shape: {ZiXR1X.shape}, stats: min={ZiXR1X.min():.6f}, max={ZiXR1X.max():.6f}")
            print(f"ZiXR0y shape: {ZiXR0y.shape}, stats: min={ZiXR0y.min():.6f}, max={ZiXR0y.max():.6f}")
            print(f"ZiXR1y shape: {ZiXR1y.shape}, stats: min={ZiXR1y.min():.6f}, max={ZiXR1y.max():.6f}")

            # Mᵀ𝕂y = Mᵀ𝓡𝐲 - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡𝐲) - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
            #       + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
            MK0y = MR0y - _mdot(XRiM.T, ZiXR0y) - _mdot(MR0X, ZiXRiy)
            MK0y += _mdot(XRiM.T, ZiXR0X, ZiXRiy)
            MK1y = MR1y - _mdot(XRiM.T, ZiXR1y) - _mdot(MR1X, ZiXRiy)
            MK1y += _mdot(XRiM.T, ZiXR1X, ZiXRiy)

            print(f"\n--- Mᵀ𝕂𝐲 Computation ---")
            print(f"MK0y shape: {MK0y.shape}, stats: min={MK0y.min():.6f}, max={MK0y.max():.6f}")
            print(f"MK1y shape: {MK1y.shape}, stats: min={MK1y.min():.6f}, max={MK1y.max():.6f}")
            print(f"  Term 1 (MR0y): {MR0y.min():.6f} to {MR0y.max():.6f}")
            print(f"  Term 2 (_mdot(XRiM.T, ZiXR0y)): {_mdot(XRiM.T, ZiXR0y).min():.6f} to {_mdot(XRiM.T, ZiXR0y).max():.6f}")
            print(f"  Term 3 (_mdot(MR0X, ZiXRiy)): {_mdot(MR0X, ZiXRiy).min():.6f} to {_mdot(MR0X, ZiXRiy).max():.6f}")
            print(f"  Term 4 (_mdot(XRiM.T, ZiXR0X, ZiXRiy)): {_mdot(XRiM.T, ZiXR0X, ZiXRiy).min():.6f} to {_mdot(XRiM.T, ZiXR0X, ZiXRiy).max():.6f}")

            # 𝐲ᵀ𝕂𝐲 = 𝐲ᵀ𝓡𝐲 - 2(𝐲ᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲) + (𝐲ᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
            yK0y = yR0y - 2 * XR0y.T @ ZiXRiy + ZiXRiy.T @ _mdot(XR0X, ZiXRiy)
            yK1y = yR1y - 2 * XR1y.T @ ZiXRiy + ZiXRiy.T @ _mdot(XR1X, ZiXRiy)

            print(f"\n--- 𝐲ᵀ𝕂𝐲 Computation ---")
            print(f"yK0y shape: {yK0y.shape}, value: {yK0y}")
            print(f"yK1y shape: {yK1y.shape}, value: {yK1y}")
            print(f"  Term 1 (yR0y): {yR0y}")
            print(f"  Term 2 (2 * XR0y.T @ ZiXRiy): {2 * XR0y.T @ ZiXRiy}")
            print(f"  Term 3 (ZiXRiy.T @ _mdot(XR0X, ZiXRiy)): {ZiXRiy.T @ _mdot(XR0X, ZiXRiy)}")

            # Mᵀ𝕂M = Mᵀ𝓡M - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹M) - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡M)
            #       + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹M)
            MR0XZiXRiM = _mdot(MR0X, ZiXRiM)
            MK0M = MR0M - MR0XZiXRiM - MR0XZiXRiM.transpose([1, 0, 2])
            MK0M += _mdot(ZiXRiM.T, XR0X, ZiXRiM)
            MR1XZiXRiM = _mdot(MR1X, ZiXRiM)
            MK1M = MR1M - MR1XZiXRiM - MR1XZiXRiM.transpose([1, 0, 2])
            MK1M += _mdot(ZiXRiM.T, XR1X, ZiXRiM)

            print(f"\n--- Mᵀ𝕂M Computation ---")
            print(f"MK0M shape: {MK0M.shape}, stats: min={MK0M.min():.6f}, max={MK0M.max():.6f}")
            print(f"MK1M shape: {MK1M.shape}, stats: min={MK1M.min():.6f}, max={MK1M.max():.6f}")
            print(f"MR0XZiXRiM shape: {MR0XZiXRiM.shape}")
            print(f"MR1XZiXRiM shape: {MR1XZiXRiM.shape}")

            # Beta-related computations
            MK0m = _mdot(MK0M, b)
            mK0y = b.T @ MK0y
            mK0m = b.T @ MK0m
            MK1m = _mdot(MK1M, b)
            mK1y = b.T @ MK1y
            mK1m = b.T @ MK1m
            XRim = XRiM @ b
            MRim = MRiM @ b

            print(f"\n--- Beta-related Terms ---")
            print(f"MK0m shape: {MK0m.shape}, stats: min={MK0m.min():.6f}, max={MK0m.max():.6f}")
            print(f"MK1m shape: {MK1m.shape}, stats: min={MK1m.min():.6f}, max={MK1m.max():.6f}")
            print(f"mK0y shape: {mK0y.shape}, value: {mK0y}")
            print(f"mK1y shape: {mK1y.shape}, value: {mK1y}")
            print(f"mK0m shape: {mK0m.shape}, value: {mK0m}")
            print(f"mK1m shape: {mK1m.shape}, value: {mK1m}")
            print(f"XRim shape: {XRim.shape}, MRim shape: {MRim.shape}")

            # ∂𝛃 computation
            db = {"C0.Lu": lu_solve(Lh, MK0m - MK0y), "C1.Lu": lu_solve(Lh, MK1m - MK1y)}

            print(f"\n--- ∂𝛃 Computation ---")
            print(f"db['C0.Lu'] shape: {db['C0.Lu'].shape}, stats: min={db['C0.Lu'].min():.6f}, max={db['C0.Lu'].max():.6f}")
            print(f"db['C1.Lu'] shape: {db['C1.Lu'].shape}, stats: min={db['C1.Lu'].min():.6f}, max={db['C1.Lu'].max():.6f}")
            print(f"MK0m - MK0y stats: min={(MK0m - MK0y).min():.6f}, max={(MK0m - MK0y).max():.6f}")
            print(f"MK1m - MK1y stats: min={(MK1m - MK1y).min():.6f}, max={(MK1m - MK1y).max():.6f}")

            # Trace terms initialization
            trace_WdC0 = trace(WdC0)
            trace_WdC1 = trace(WdC1)
            trace_ZiXR0X = trace(ZiXR0X)
            trace_ZiXR1X = trace(ZiXR1X)

            grad = {
                "C0.Lu": -trace_WdC0 * self._trGG + trace_ZiXR0X,
                "C1.Lu": -trace_WdC1 * self.nsamples + trace_ZiXR1X,
            }

            print(f"\n--- Initial Trace Terms ---")
            print(f"trace(WdC0): {trace_WdC0}")
            print(f"trace(WdC1): {trace_WdC1}")
            print(f"trace(ZiXR0X): {trace_ZiXR0X}")
            print(f"trace(ZiXR1X): {trace_ZiXR1X}")
            print(f"self._trGG: {self._trGG}, self.nsamples: {self.nsamples}")
            print(f"Initial grad['C0.Lu']: {grad['C0.Lu']}")
            print(f"Initial grad['C1.Lu']: {grad['C1.Lu']}")

            # Restricted case handling
            if self._restricted:
                restricted_term_C0 = lu_solve(Lh, MK0M).diagonal().sum(1)
                restricted_term_C1 = lu_solve(Lh, MK1M).diagonal().sum(1)
                grad["C0.Lu"] += restricted_term_C0
                grad["C1.Lu"] += restricted_term_C1

                print(f"\n--- Restricted Terms ---")
                print(f"Restricted term C0: {restricted_term_C0}")
                print(f"Restricted term C1: {restricted_term_C1}")
                print(f"Updated grad['C0.Lu']: {grad['C0.Lu']}")
                print(f"Updated grad['C1.Lu']: {grad['C1.Lu']}")

            # Final gradient terms
            mKiM = MRim.T - XRim.T @ ZiXRiM
            yKiM = MRiy.T - XRiy.T @ ZiXRiM

            print(f"\n--- Final Gradient Terms Setup ---")
            print(f"mKiM shape: {mKiM.shape}, stats: min={mKiM.min():.6f}, max={mKiM.max():.6f}")
            print(f"yKiM shape: {yKiM.shape}, stats: min={yKiM.min():.6f}, max={yKiM.max():.6f}")
            print(f"MRim.T shape: {MRim.T.shape}, XRim.T @ ZiXRiM shape: {(XRim.T @ ZiXRiM).shape}")

            # Final gradient assembly
            print(f"\n--- Final Gradient Assembly ---")
            print(f"Before final terms - grad['C0.Lu']: {grad['C0.Lu']}")
            print(f"Before final terms - grad['C1.Lu']: {grad['C1.Lu']}")

            term1_C0 = yK0y
            term2_C0 = -2 * mK0y
            term3_C0 = mK0m
            term4_C0 = -2 * _mdot(mKiM, db["C0.Lu"])
            term5_C0 = 2 * _mdot(yKiM, db["C0.Lu"])

            term1_C1 = yK1y
            term2_C1 = -2 * mK1y
            term3_C1 = mK1m
            term4_C1 = -2 * _mdot(mKiM, db["C1.Lu"])
            term5_C1 = 2 * _mdot(yKiM, db["C1.Lu"])

            print(f"C0 gradient terms:")
            print(f"  Term 1 (yK0y): {term1_C0}")
            print(f"  Term 2 (-2 * mK0y): {term2_C0}")
            print(f"  Term 3 (mK0m): {term3_C0}")
            print(f"  Term 4 (-2 * _mdot(mKiM, db['C0.Lu'])): {term4_C0}")
            print(f"  Term 5 (2 * _mdot(yKiM, db['C0.Lu'])): {term5_C0}")

            print(f"C1 gradient terms:")
            print(f"  Term 1 (yK1y): {term1_C1}")
            print(f"  Term 2 (-2 * mK1y): {term2_C1}")
            print(f"  Term 3 (mK1m): {term3_C1}")
            print(f"  Term 4 (-2 * _mdot(mKiM, db['C1.Lu'])): {term4_C1}")
            print(f"  Term 5 (2 * _mdot(yKiM, db['C1.Lu'])): {term5_C1}")

            grad["C0.Lu"] += term1_C0 + term2_C0 + term3_C0 + term4_C0 + term5_C0
            grad["C1.Lu"] += term1_C1 + term2_C1 + term3_C1 + term4_C1 + term5_C1

            print(f"\nAfter adding all terms:")
            print(f"grad['C0.Lu']: {grad['C0.Lu']}")
            print(f"grad['C1.Lu']: {grad['C1.Lu']}")

            grad["C0.Lu"] /= 2
            grad["C1.Lu"] /= 2

            print(f"\nFinal gradients (after /= 2):")
            print(f"grad['C0.Lu']: {grad['C0.Lu']}")
            print(f"grad['C1.Lu']: {grad['C1.Lu']}")

            print("Final C0: ", self._cov.C0.value())
            print("Final C1: ", self._cov.C1.value())
            
            print("=== LML GRADIENT COMPUTATION END ===\n")

            return grad
    
    @cached_property
    def _logdet_MM(self):
        if not self._restricted:
            return 0.0

        M = self._mean.AX
        ldet = slogdet(M.T @ M)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of MᵀM should be positive.")
        return ldet[1]

    @property
    def _logdetH(self):
        if not self._restricted:
            return 0.0
        terms = self._terms
        MKiM = terms["MRiM"] - terms["XRiM"].T @ terms["ZiXRiM"]
        return slogdet(MKiM)[1]

    @property
    def _logdetK(self):
        from numpy_sugar.linalg import lu_slogdet

        terms = self._terms
        S = terms["S"]
        Lz = terms["Lz"]

        cov_logdet = lu_slogdet(Lz)[1]
        cov_logdet -= 2 * log(S).sum() * self.nsamples
        return cov_logdet

    @property
    def _df(self):
        np = self.nsamples * self.ntraits
        if not self._restricted:
            return np
        cp = self.ncovariates * self.ntraits
        return np - cp


def _dot(a, b):
    r = tensordot(a, b, axes=([min(1, a.ndim - 1)], [0]))
    if a.ndim > b.ndim:
        if r.ndim == 3:
            return r.transpose([0, 2, 1])
        return r
    return r


def _mdot(*args):
    return reduce(_dot, args)


def _mkron(a, b):
    if a.ndim == 3:
        return kron(a.transpose([2, 0, 1]), b).transpose([1, 2, 0])
    return kron(a, b)

#Unitil here!!!!!!!!!!!!!------------------------------------------------


class VarDec(object):
    """
    Variance decompositon through GLMMs.

    Example
    -------

    .. doctest::

        >>> from limix.vardec import VarDec
        >>> from limix.stats import multivariate_normal as mvn
        >>> from numpy import ones, eye, concatenate, zeros, exp
        >>> from numpy.random import RandomState
        >>>
        >>> random = RandomState(0)
        >>> nsamples = 20
        >>>
        >>> M = random.randn(nsamples, 2)
        >>> M = (M - M.mean(0)) / M.std(0)
        >>> M = concatenate((ones((nsamples, 1)), M), axis=1)
        >>>
        >>> K0 = random.randn(nsamples, 10)
        >>> K0 = K0 @ K0.T
        >>> K0 /= K0.diagonal().mean()
        >>> K0 += eye(nsamples) * 1e-4
        >>>
        >>> K1 = random.randn(nsamples, 10)
        >>> K1 = K1 @ K1.T
        >>> K1 /= K1.diagonal().mean()
        >>> K1 += eye(nsamples) * 1e-4
        >>>
        >>> y = M @ random.randn(3) + mvn(random, zeros(nsamples), K0)
        >>> y += mvn(random, zeros(nsamples), K1)
        >>>
        >>> vardec = VarDec(y, "normal", M)
        >>> vardec.append(K0)
        >>> vardec.append(K1)
        >>> vardec.append_iid()
        >>>
        >>> vardec.fit(verbose=False)
        >>> print(vardec) # doctest: +FLOAT_CMP
        Variance decomposition
        ----------------------
        <BLANKLINE>
        𝐲 ~ 𝓝(𝙼𝜶, 0.385⋅𝙺 + 1.184⋅𝙺 + 0.000⋅𝙸)
        >>> y = exp((y - y.mean()) / y.std())
        >>> vardec = VarDec(y, "poisson", M)
        >>> vardec.append(K0)
        >>> vardec.append(K1)
        >>> vardec.append_iid()
        >>>
        >>> vardec.fit(verbose=False)
        >>> print(vardec) # doctest: +FLOAT_CMP
        Variance decomposition
        ----------------------
        <BLANKLINE>
        𝐳 ~ 𝓝(𝙼𝜶, 0.000⋅𝙺 + 0.350⋅𝙺 + 0.000⋅𝙸) for yᵢ ~ Poisson(λᵢ=g(zᵢ)) and g(x)=eˣ
    """

    def __init__(self, y, lik="normal", M=None):
        """
        Constructor.

        Parameters
        ----------
        y : array_like
            Phenotype.
        lik : tuple, "normal", "bernoulli", "probit", "binomial", "poisson"
            Sample likelihood describing the residual distribution.
            Either a tuple or a string specifying the likelihood is required. The
            Normal, Bernoulli, Probit, and Poisson likelihoods can be selected by
            providing a string. Binomial likelihood on the other hand requires a tuple
            because of the number of trials: ``("binomial", array_like)``. Defaults to
            ``"normal"``.
        M : n×c array_like
            Covariates matrix.
        """
        from numpy import asarray, eye
        from glimix_core.mean import LinearMean, KronMean

        y = asarray(y, float)
        data = conform_dataset(y, M)
        y = data["y"]
        M = data["M"]
        self._y = y
        self._M = M
        self._lik = normalize_likelihood(lik)
        if self._multi_trait():
            A = eye(self._y.shape[1])
            self._mean = KronMean(A, asarray(M, float))
        else:
            self._mean = LinearMean(asarray(M, float))
        self._covariance = []
        self._glmm = None
        self._fit = False
        self._unnamed = 0

    @property
    def effsizes(self):
        """
        Covariance effect sizes.

        Returns
        -------
        effsizes : ndarray
            Effect sizes.
        """
        if not self._fit:
            self.fit()
        if hasattr(self._mean, "effsizes"):
            return self._mean.effsizes
        return self._mean.B

    @property
    def covariance(self):
        """
        Get the covariance matrices.

        Returns
        -------
        covariances : list
            Covariance matrices.
        """
        return self._covariance

    def fit(self, verbose=True):
        """
        Fit the model.

        Parameters
        ----------
        verbose : bool, optional
            Set ``False`` to silence it. Defaults to ``True``.
        """
        with session_block("Variance decomposition", disable=not verbose):
            if self._lik[0] == "normal":
                #changed!
                if self._multi_trait(): 
                    if self._simple_model():
                        self._fit_lmm_multi_trait_simple(verbose)
                    #else:
                    #    self._fit_lmm(verbose)       
                elif self._simple_model():
                    self._fit_lmm_simple_model(verbose)
                else:
                    self._fit_lmm(verbose)
            else:
                if self._simple_model():
                    self._fit_glmm_simple_model(verbose)
                else:
                    self._fit_glmm(verbose)

            if verbose:
                print(self)

        self._fit = True

    def lml(self):
        """
        Get the log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        if not self._fit:
            self._glmm.fit()
        return self._glmm.lml()

    def append_iid(self, name="residual"):
        from glimix_core.cov import EyeCov

        if self._multi_trait():
            cov = MTEyeCov(self._y.shape[1])
        else:
            cov = EyeCov(self._y.shape[0])

        cov.name = name
        self._covariance.append(cov)

    def append(self, K, name=None):
        from numpy_sugar import is_all_finite
        from numpy import asarray
        from glimix_core.cov import GivenCov

        data = conform_dataset(self._y, K=K)
        K = asarray(data["K"], float)

        if not is_all_finite(K):
            raise ValueError("Covariance-matrix values must be finite.")

        K = K / K.diagonal().mean()
        if self._multi_trait():
            #print('test name: ', name)
            cov = MTGivenCov(self._y.shape[1], K)

        else:
            #print('test name: ', name)
            cov = GivenCov(K)

        if name is None:
            name = "unnamed-{}".format(self._unnamed)
            self._unnamed += 1
        cov.name = name

        self._covariance.append(cov)
    
    def plot(self):
        from limix_modified.plot._plt import get_pyplot
        from limix_modified.plot._show import show
        import seaborn as sns
        from matplotlib.ticker import FormatStrFormatter

        variances = [c.scale for c in self._covariance]
        variances = [(v / sum(variances)) * 100 for v in variances]
        names = [c.name for c in self._covariance]

        ax = sns.barplot(x=names, y=variances)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f%%"))
        ax.set_xlabel("random effects")
        ax.set_ylabel("explained variance")
        ax.set_title("Variance decomposition")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_pyplot().tight_layout()
        show()
     
    # performed if K_cis and K_trans are handed over 
    def _fit_lmm(self, verbose):
        #print("Fit LMM. Not simple model.")
        from glimix_core.cov import SumCov
        from glimix_core.gp import GP
        from numpy import asarray

        y = asarray(self._y, float).ravel()
        gp = GP(y, self._mean, SumCov(self._covariance))
        #print('print covariances: ', SumCov(self._covariance))
        gp.fit(verbose=verbose)
        self._glmm = gp

    def _fit_glmm(self, verbose):
        from glimix_core.cov import SumCov
        from glimix_core.ggp import ExpFamGP
        from numpy import asarray

        y = asarray(self._y, float).ravel()
        gp = ExpFamGP(y, self._lik, self._mean, SumCov(self._covariance))
        #print("Covariance before fitting:", self._covariance)
        gp.fit(verbose=verbose)
        #print("Covariance after fitting:", self._covariance)
        self._glmm = gp

    def _fit_lmm_multi_trait_simple(self, verbose):
        from numpy import sqrt, asarray
        #from glimix_core.lmm import Kron2Sum
        from numpy_sugar.linalg import economic_qs, ddot
        #print("Fit simple multi trait model.")
        X = asarray(self._M, float)

        QS = economic_qs(self._covariance[0]._K)
        G = ddot(QS[0][0], sqrt(QS[1]))
        lmm = Kron2Sum(self._y, self._mean.A, X, G, rank=1, restricted=True)
        lmm.fit(verbose=verbose)
        self._glmm = lmm
        self._covariance[0]._set_kron2sum(lmm)
        self._covariance[1]._set_kron2sum(lmm)
        self._mean.B = lmm.B

    
    def _fit_lmm_simple_model(self, verbose):
        from numpy_sugar.linalg import economic_qs
        from glimix_core.lmm import LMM
        from numpy import asarray
        print('Fit simple model.')
        K = self._get_matrix_simple_model()

        y = asarray(self._y, float).ravel()
        QS = None
        if K is not None:
            QS = economic_qs(K)
        lmm = LMM(y, self._M, QS)
        lmm.fit(verbose=verbose)
        self._set_simple_model_variances(lmm.v0, lmm.v1)
        self._glmm = lmm

    def _fit_glmm_simple_model(self, verbose):
        from numpy_sugar.linalg import economic_qs
        from glimix_core.glmm import GLMMExpFam
        from numpy import asarray

        K = self._get_matrix_simple_model()

        y = asarray(self._y, float).ravel()
        QS = None
        if K is not None:
            QS = economic_qs(K)

        glmm = GLMMExpFam(y, self._lik, self._M, QS)
        glmm.fit(verbose=verbose)

        self._set_simple_model_variances(glmm.v0, glmm.v1)
        self._glmm = glmm

    def _set_simple_model_variances(self, v0, v1):
        from glimix_core.cov import GivenCov, EyeCov

        for c in self._covariance:
            if isinstance(c, GivenCov):
                c.scale = v0
            elif isinstance(c, EyeCov):
                c.scale = v1

    def _get_matrix_simple_model(self):
        from glimix_core.cov import GivenCov

        K = None
        for i in range(len(self._covariance)):
            #print("test index: ", i)
            if isinstance(self._covariance[i], GivenCov):
                self._covariance[i].scale = 1.0
                K = self._covariance[i].value()
                #print('K: ', K)
                break
        return K

    def _multi_trait(self):
        return self._y.ndim == 2 and self._y.shape[1] > 1

    def _simple_model(self):
        from glimix_core.cov import GivenCov, EyeCov

        if len(self._covariance) > 2:
            return False

        c = self._covariance
        if len(c) == 1 and isinstance(c[0], EyeCov):
            return True

        if isinstance(c[0], GivenCov) and isinstance(c[1], EyeCov):
            return True

        if isinstance(c[1], GivenCov) and isinstance(c[0], EyeCov):
            return True
        
        if isinstance(c[0], MTGivenCov) and isinstance(c[1], MTEyeCov):
            return True

        if isinstance(c[1], MTGivenCov) and isinstance(c[0], MTEyeCov):
            return True

        return False
    
    def __repr__(self):
        from glimix_core.cov import GivenCov
        from limix.qtl._result._draw import draw_model
        from limix._display import draw_title
        import numpy as np

        covariance = ""
        for c in self._covariance:
            # Get the scale
            s = c.scale
    
            # If it's a matrix (multi-trait case), summarize it
            if hasattr(s, "shape") and len(s.shape) == 2:
                if s.shape[0] == s.shape[1]:  # square matrix
                    s = s.diagonal().mean()
                else:
                    s = s.mean()  # fallback for non-square, should not happen in trait covariance
    
            # Format the covariance string
            if isinstance(c, GivenCov):
                covariance += f"{s:.3f}⋅𝙺 + "
            else:
                covariance += f"{s:.3f}⋅𝙸 + "
    
        # Remove the trailing " + " if present
        if covariance.endswith(" + "):
            covariance = covariance[:-3]
    
        return f"VarDec({covariance})"
    
class MTGivenCov:
    def __init__(self, ntraits, K):
        self._ntraits = ntraits
        self._K = K
        self._kron2sum = None
        self._name = "unnamed"

    def _set_kron2sum(self, kron2sum):
        self._kron2sum = kron2sum

    @property
    def scale(self):
        """
        Scale parameter, s.
        """
        from numpy import eye

        if self._kron2sum is None:
            return eye(self._ntraits)
        return self._kron2sum.C0

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


class MTEyeCov:
    def __init__(self, ntraits):
        self._ntraits = ntraits
        self._kron2sum = None
        self._name = "unnamed"

    def _set_kron2sum(self, kron2sum):
        self._kron2sum = kron2sum

    @property
    def scale(self):
        """
        Scale parameter, s.
        """
        from numpy import eye

        if self._kron2sum is None:
            return eye(self._ntraits)
        return self._kron2sum.C1

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


    

    '''

        
    def __repr__(self):
        from glimix_core.cov import GivenCov
        from limix_modified.qtl.result._draw import draw_model
        from limix_modified._display._draw import draw_title

        covariance = ""
        for c in self._covariance:
            print("all covariances: ", self._covariance)
            print('test c: ', c)
            s = c.scale
            print('test s: ', s)
            if isinstance(c, GivenCov):
                # For single-trait models
                covariance += "{:.3f}⋅𝙺 + ".format(s)

            elif isinstance(c, MTGivenCov):
                # For multi-trait models, handle matrix scale
                diag_elements = np.diag(s)  # Extract diagonal (genetic variance)
                off_diag_elements = s[np.triu_indices_from(s, k=1)]  # Extract off-diagonal (GxE)
                covariance += "Diag: [{}]⋅𝙺 + Off-diag: [{}]⋅𝙺 + ".format(
                    ", ".join(f"{v:.3f}" for v in diag_elements),
                    ", ".join(f"{v:.3f}" for v in off_diag_elements),
                )
            else:
                # For other types of covariance (e.g., identity matrix)
                covariance += "{:.3f}⋅𝙸 + ".format(np.mean(s))

        msg = draw_title("Variance decomposition")
        msg += draw_model(self._lik[0], "𝙼𝜶", covariance[:-3])  # Remove trailing "+ "
        msg = msg.rstrip()

        return msg

    def __repr__(self):
        import numpy as np
        from glimix_core.cov import GivenCov
        from limix_modified.qtl.result._draw import draw_model
        from limix_modified._display._draw import draw_title

        covariance = ""
        if self._multi_trait():
            # Handle multi-trait case
            for c in self._covariance:
                if hasattr(c.scale, 'trace'):  # If scale is a matrix, use its trace
                    trace_value = np.trace(c.scale)
                    covariance += "{:.3f}⋅𝙺 + ".format(trace_value)
                else:
                    covariance += "{:.3f}⋅𝙺 + ".format(c.scale)
        else:
            # Handle single-trait case
            for c in self._covariance:
                s = c.scale
                if isinstance(c, GivenCov):
                    covariance += "{:.3f}⋅𝙺 + ".format(s)
                else:
                    covariance += "{:.3f}⋅𝙸 + ".format(s)

        if len(covariance) > 2:
            covariance = covariance[:-3]  # Remove the trailing "+"

        # Generate the title and model representation
        msg = draw_title("Variance decomposition")
        msg += draw_model(self._lik[0], "𝙼𝜶", covariance)
        msg = msg.rstrip()

        return msg
    

    def __repr__(self):
        from glimix_core.cov import GivenCov
        from limix_modified.qtl.result._draw import draw_model
        from limix_modified._display._draw import draw_title

        if self._multi_trait():
            # Multi-trait case: Represent multiple traits with different covariance structures
            covariance = ""
            for c in self._covariance:
                scales = c.scale  # In multi-trait, scale is a matrix, so we'll display its diagonal
                diagonal_scales = scales.diagonal()     
                # DEBUG: Check the type of covariance component
                print(f"Component: {c}, Type: {type(c)}")
                if isinstance(c, MTGivenCov):
                    covariance += "[" + ", ".join("{:.3f}".format(s) for s in diagonal_scales) + "]⋅𝙺 + "
                else:
                    covariance += "[" + ", ".join("{:.3f}".format(s) for s in diagonal_scales) + "]⋅𝙸 + "
            if len(covariance) > 2:
                covariance = covariance[:-3]  # Remove trailing ' + '

            # Representation for multi-trait variance decomposition
            msg = draw_title("Multi-Trait Variance Decomposition")
            msg += draw_model(self._lik[0], "𝙼𝜶", covariance)
        else:
            # Single-trait case: Use the existing logic for single-trait models
            covariance = ""
            for c in self._covariance:
                s = c.scale
                if isinstance(c, GivenCov):
                    covariance += "{:.3f}⋅𝙺 + ".format(s)
                else:
                    covariance += "{:.3f}⋅𝙸 + ".format(s)
            if len(covariance) > 2:
                covariance = covariance[:-3]  # Remove trailing ' + '
            # Representation for single-trait variance decomposition
            msg = draw_title("Variance decomposition")
            msg += draw_model(self._lik[0], "𝙼𝜶", covariance)

        msg = msg.rstrip()
        return msg
    '''

    '''
    def plot(self):
        from limix_modified.plot._plt import get_pyplot
        from limix_modified.plot._show import show
        import seaborn as sns
        from matplotlib.ticker import FormatStrFormatter

        if self._multi_trait():
            # For multi-trait, use trace or another summary statistic for variances
            variances = []
            for c in self._covariance:
                if hasattr(c.scale, 'trace'):
                    trace_value = c.scale.trace()
                    variances.append(trace_value)
                    print(f"{c.name}: Trace: {trace_value}")
                else:
                    variances.append(c.scale)
                    print(f"{c.name}: Scale: {c.scale}")
        else:
            # Single trait
            variances = [c.scale for c in self._covariance]

        total_variance = sum(variances)
        if total_variance == 0:
            raise ValueError("Sum of variances is zero, cannot compute percentages.")
        variances = [(v / total_variance) * 100 for v in variances]
        names = [c.name for c in self._covariance]

        # Debugging information
        print(f"Variances: {variances}")
        print(f"Names: {names}")

        ax = sns.barplot(x=names, y=variances)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f%%"))
        ax.set_xlabel("Random Effects")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Variance Decomposition")

        # Adjust layout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_pyplot().tight_layout()
        
        show()
    '''

'''
    def _fit_lmm_multi_trait(self, verbose):
        from numpy import sqrt, asarray
        from glimix_core.lmm import Kron2Sum
        from numpy_sugar.linalg import economic_qs, ddot

        print("Fit simple multi-trait model.")
        X = asarray(self._M, float)
        
        #Changed!
        # Loop over covariance matrices to create separate Kron2Sum instances
        for c in self._covariance:
            print('testcov in _covariance: ', c)
            print('type of c: ', type(c))
            print('dict of c: ', c.__dict__)


            if isinstance(c, MTGivenCov):
                QS = economic_qs(c._K)  # Covariance-specific QS decomposition
                G = ddot(QS[0][0], sqrt(QS[1]))

            elif isinstance(c, MTEyeCov):
                c_K = None
                G = None

            # Create Kron2Sum instance for MTGivenCov
            lmm = Kron2Sum(self._y, self._mean.A, X, G, rank=1, restricted=True)
            lmm.fit(verbose=verbose)
            self._glmm = lmm

            if isinstance(c, MTGivenCov):
                # Assign Kron2Sum to the corresponding covariance object
                c._set_kron2sum(lmm)

            elif isinstance(c, MTEyeCov):
                # Handle MTEyeCov case (identity matrix)
                print("Processing MTEyeCov (identity covariance).")

                c._set_kron2sum
                print('MTEye kron2sum: ', c._set_kron2sum)

                # Assign Kron2Sum back to MTEyeCov if needed (though it doesn't have _set_kron2sum)
                # You may need to define a method in MTEyeCov for setting the Kron2Sum if required.
                # For now, we will skip that part for MTEyeCov.

            else:
                raise ValueError(f"Unsupported covariance type: {type(c)}")

            print('Updated dict after setting Kron2Sum: ', c.__dict__)

            # Set the effect sizes specific to the covariance matrix
            self._mean.B = lmm.B  # Effect sizes

            #QS = economic_qs(self._covariance[0]._K)
            #G = ddot(QS[0][0], sqrt(QS[1]))
            #lmm = Kron2Sum(self._y, self._mean.A, X, G, rank=1, restricted=True)
            #lmm.fit(verbose=verbose)
            #self._glmm = lmm
            #self._covariance[0]._set_kron2sum(lmm) # extract the covariances from the LMM fit
            #self._covariance[1]._set_kron2sum(lmm)
            #self._mean.B = lmm.B # effect sizes
    '''
