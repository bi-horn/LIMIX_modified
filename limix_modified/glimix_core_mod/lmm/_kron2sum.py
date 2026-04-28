'''Copied 1-to-1 from original implementation:
https://github.com/limix/glimix-core/blob/master/glimix_core/lmm/_kron2sum.py
Only changed Function and Kron2SumCov import for refactored verbose prints and different lrfreeform covariance initialization
'''

import warnings
from functools import reduce

from numpy import asarray, asfortranarray, kron, log, sqrt, tensordot, trace
from numpy.linalg import inv, matrix_rank, slogdet

from limix_modified.glimix_core_mod.cov._kron2sum import Kron2SumCov
from limix_modified.optimix_mod.optimix import Function # instead of: from optimix import Function to enable stdout

from glimix_core._util import cached_property, log2pi, unvec, vec
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

    def __init__(self, Y, A, X, G, rank, original_L_init=False, restricted=False):
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
        original_L_init: optional, bool
            Originally C0 was initialized by only ones leading to rank-defficient C0; QR initialization (original_L_init=False) was found to perform superior    
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
        self._cov = Kron2SumCov(G, Y.shape[1], rank, original_L_init)
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
        terms = self._terms
        yKiy = terms["yKiy"]
        mKiy = terms["mKiy"]
        mKim = terms["mKim"]

        lml = -self._df * log2pi + self._logdet_MM - self._logdetK
        lml -= self._logdetH
        lml += -yKiy - mKim + 2 * mKiy

        return lml / 2

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
        """
        from numpy_sugar.linalg import lu_solve
        import numpy as np

        terms = self._terms
        dC0 = self._cov.C0.gradient()["Lu"]
        dC1 = self._cov.C1.gradient()["Lu"]

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

        WdC0 = _mdot(W, dC0)
        WdC1 = _mdot(W, dC1)

        AWdC0 = _mdot(WA.T, dC0)
        AWdC1 = _mdot(WA.T, dC1)
        AWdC0_WA = _mdot(AWdC0, WA)
        AWdC1_WA = _mdot(AWdC1, WA)

        # Mᵀ𝓡M
        MR0M = _mkron(AWdC0_WA, self._XGGX)
        MR1M = _mkron(AWdC1_WA, self._XX)

        # Mᵀ𝓡X
        AWdC0_WL0 = _mdot(AWdC0, WL0)
        AWdC1_WL0 = _mdot(AWdC1, WL0)
        MR0X = _mkron(AWdC0_WL0, self._XGGG)
        MR1X = _mkron(AWdC1_WL0, self._GX.T)

        # Mᵀ𝓡𝐲 = (AᵀW∂Cᵢ⊗XᵀEᵢ)vec(EᵢᵀYW) = vec(XᵀEᵢEᵢᵀYW∂CᵢWA)
        WdC0_WA = _mdot(WdC0, WA)
        XGGY_WdC0_WA = _mdot(self._XGGY, WdC0_WA)
        MR0y = vec(XGGY_WdC0_WA)
        MR1y = vec(_mdot(self._XY, WdC1, WA))

        # Xᵀ𝓡X
        WL0T_dC0_WL0 = _mdot(WL0.T, dC0, WL0)
        WL0T_dC1_WL0 = _mdot(WL0.T, dC1, WL0)
        XR0X = _mkron(WL0T_dC0_WL0, self._GGGG)
        XR1X = _mkron(WL0T_dC1_WL0, self._GG)

        # Xᵀ𝓡𝐲
        GGGY_WdC0_WL0 = _mdot(self._GGGY, WdC0, WL0)
        XR0y = vec(GGGY_WdC0_WL0)
        XR1y = vec(_mdot(self._GY, WdC1, WL0))

        # 𝐲ᵀ𝓡𝐲 = vec(EᵢᵀYW∂Cᵢ)ᵀvec(EᵢᵀYW)
        GY_WdC0 = _mdot(self._GY, WdC0)
        GY_W = self._GY @ W
        yR0y = vec(GY_WdC0).T @ vec(GY_W)

        yR1y = (YW.T * _mdot(self._Y, WdC1).T).T.sum(axis=(0, 1))

        ZiXR0X = lu_solve(Lz, XR0X)
        ZiXR1X = lu_solve(Lz, XR1X)
        ZiXR0y = lu_solve(Lz, XR0y)
        ZiXR1y = lu_solve(Lz, XR1y)

        # Mᵀ𝕂y
        MK0y = MR0y - _mdot(XRiM.T, ZiXR0y) - _mdot(MR0X, ZiXRiy)
        MK0y += _mdot(XRiM.T, ZiXR0X, ZiXRiy)
        MK1y = MR1y - _mdot(XRiM.T, ZiXR1y) - _mdot(MR1X, ZiXRiy)
        MK1y += _mdot(XRiM.T, ZiXR1X, ZiXRiy)

        # 𝐲ᵀ𝕂𝐲
        XR0y_T_ZiXRiy = XR0y.T @ ZiXRiy
        ZiXRiy_XR0X_ZiXRiy = ZiXRiy.T @ _mdot(XR0X, ZiXRiy)
        yK0y = yR0y - 2 * XR0y_T_ZiXRiy + ZiXRiy_XR0X_ZiXRiy

        XR1y_T_ZiXRiy = XR1y.T @ ZiXRiy
        ZiXRiy_XR1X_ZiXRiy = ZiXRiy.T @ _mdot(XR1X, ZiXRiy)
        yK1y = yR1y - 2 * XR1y_T_ZiXRiy + ZiXRiy_XR1X_ZiXRiy

        # Mᵀ𝕂M
        MR0XZiXRiM = _mdot(MR0X, ZiXRiM)
        MK0M = MR0M - MR0XZiXRiM - MR0XZiXRiM.transpose([1, 0, 2])
        MK0M += _mdot(ZiXRiM.T, XR0X, ZiXRiM)
        MR1XZiXRiM = _mdot(MR1X, ZiXRiM)
        MK1M = MR1M - MR1XZiXRiM - MR1XZiXRiM.transpose([1, 0, 2])
        MK1M += _mdot(ZiXRiM.T, XR1X, ZiXRiM)

        MK0m = _mdot(MK0M, b)
        mK0y = b.T @ MK0y
        mK0m = b.T @ MK0m
        MK1m = _mdot(MK1M, b)
        mK1y = b.T @ MK1y
        mK1m = b.T @ MK1m
        XRim = XRiM @ b
        MRim = MRiM @ b

        db = {"C0.Lu": lu_solve(Lh, MK0m - MK0y), "C1.Lu": lu_solve(Lh, MK1m - MK1y)}

        trace_WdC0 = trace(WdC0)
        trace_WdC1 = trace(WdC1)
        trace_ZiXR0X = trace(ZiXR0X)
        trace_ZiXR1X = trace(ZiXR1X)

        grad = {
            "C0.Lu": -trace_WdC0 * self._trGG + trace_ZiXR0X,
            "C1.Lu": -trace_WdC1 * self.nsamples + trace_ZiXR1X,
        }

        if self._restricted:
            HiMK0M = lu_solve(Lh, MK0M)
            HiMK1M = lu_solve(Lh, MK1M)
            trace_HiMK0M = HiMK0M.diagonal().sum(1)
            trace_HiMK1M = HiMK1M.diagonal().sum(1)

            print(f"\n[Step 17: Restricted correction]")
            print(f"  trace(H^-1 @ MK0M): {trace_HiMK0M}")
            print(f"  trace(H^-1 @ MK1M): {trace_HiMK1M}")

            grad["C0.Lu"] += trace_HiMK0M
            grad["C1.Lu"] += trace_HiMK1M

            print(f"  grad['C0.Lu'] (after restricted): {grad['C0.Lu']}")
            print(f"  grad['C1.Lu'] (after restricted): {grad['C1.Lu']}")
    
        mKiM = MRim.T - XRim.T @ ZiXRiM
        yKiM = MRiy.T - XRiy.T @ ZiXRiM

        mKiM_db_C0 = _mdot(mKiM, db["C0.Lu"])
        yKiM_db_C0 = _mdot(yKiM, db["C0.Lu"])
        mKiM_db_C1 = _mdot(mKiM, db["C1.Lu"])
        yKiM_db_C1 = _mdot(yKiM, db["C1.Lu"])

        db_contribution_C0 = -2 * mKiM_db_C0 + 2 * yKiM_db_C0
        db_contribution_C1 = -2 * mKiM_db_C1 + 2 * yKiM_db_C1

        c0_terms = yK0y - 2 * mK0y + mK0m
        c1_terms = yK1y - 2 * mK1y + mK1m

        grad["C0.Lu"] += yK0y - 2 * mK0y + mK0m - 2 * mKiM_db_C0 + 2 * yKiM_db_C0
        grad["C1.Lu"] += yK1y - 2 * mK1y + mK1m - 2 * mKiM_db_C1 + 2 * yKiM_db_C1

        grad["C0.Lu"] /= 2
        grad["C1.Lu"] /= 2

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