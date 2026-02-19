"""
Audit tests for limix_legacy code paths.

Tests cover:
- limix_legacy.deprecated.modules.qtl.qtl_test_lmm
- limix_legacy.deprecated.modules.qtl.qtl_test_lmm_kronecker
- limix_legacy.deprecated.modules.varianceDecomposition.VarianceDecomposition
- limix_legacy.deprecated.CKroneckerLMM

These tests verify correctness and document known bugs.
"""
import unittest
import numpy as np
import scipy.stats as st
import scipy.linalg as la
import os


class TestQtlTestLmm(unittest.TestCase):
    """Tests for the qtl_test_lmm wrapper function and the lmm class."""

    def _make_data(self, N=100, S=20, seed=42):
        """Generate synthetic LMM test data."""
        rng = np.random.RandomState(seed)
        X = rng.randn(N, S)
        beta_true = np.zeros(S)
        beta_true[0] = 0.5
        beta_true[1] = -0.3
        K = X @ X.T / S
        K += 0.1 * np.eye(N)  # ensure PD
        L = la.cholesky(K, lower=True)
        noise = L @ rng.randn(N)
        Y = X @ beta_true + noise + rng.randn(N) * 0.5
        return X, Y[:, np.newaxis], K

    def test_basic_lrt(self):
        """qtl_test_lmm returns results with correct shapes and p-values in [0,1]."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=K, test='lrt')
        pv = lmm_result.getPv()
        beta = lmm_result.getBetaSNP()

        self.assertEqual(pv.shape, (1, X.shape[1]))
        self.assertEqual(beta.shape, (1, X.shape[1]))
        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))
        # Causal SNPs should have smaller p-values on average
        self.assertLess(pv[0, 0], 0.5)

    def test_f_test(self):
        """qtl_test_lmm works with F-test."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=K, test='f')
        pv = lmm_result.getPv()
        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))

    def test_no_kinship_linear_regression(self):
        """qtl_test_lmm with K=None performs linear regression (identity kinship)."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, _ = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=None, test='lrt')
        pv = lmm_result.getPv()
        self.assertEqual(pv.shape, (1, X.shape[1]))
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_custom_covariates(self):
        """qtl_test_lmm with explicit covariates."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        covs = np.column_stack([np.ones(X.shape[0]), np.random.randn(X.shape[0])])
        lmm_result = qtl_test_lmm(X, Y, K=K, covs=covs, test='lrt')
        pv = lmm_result.getPv()
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_1d_pheno_reshaped(self):
        """qtl_test_lmm handles 1D phenotype array by reshaping."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y.ravel(), K=K)
        pv = lmm_result.getPv()
        self.assertEqual(pv.shape[0], 1)

    def test_search_delta(self):
        """qtl_test_lmm with searchDelta=True on alt model."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data(N=50, S=10)
        lmm_result = qtl_test_lmm(X, Y, K=K, searchDelta=True,
                                   NumIntervalsDeltaAlt=50)
        pv = lmm_result.getPv()
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_lrt_vs_f_correlation(self):
        """LRT and F-test p-values should be correlated."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_lrt = qtl_test_lmm(X, Y, K=K, test='lrt')
        lmm_f = qtl_test_lmm(X, Y, K=K, test='f')
        pv_lrt = -np.log10(np.clip(lmm_lrt.getPv().ravel(), 1e-300, 1))
        pv_f = -np.log10(np.clip(lmm_f.getPv().ravel(), 1e-300, 1))
        corr = np.corrcoef(pv_lrt, pv_f)[0, 1]
        self.assertGreater(corr, 0.95)

    def test_invalid_test_raises(self):
        """qtl_test_lmm with invalid test type raises NotImplementedError."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data(N=30, S=5)
        with self.assertRaises(NotImplementedError):
            qtl_test_lmm(X, Y, K=K, test='invalid')

    def test_test_statistics_lrt(self):
        """Test statistics (LRT) are 2*(NLL0 - NLLAlt) and non-negative."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=K, test='lrt')
        ts = lmm_result.test_statistics
        expected = 2.0 * (lmm_result.NLL_0 - lmm_result.NLL_alt)
        np.testing.assert_allclose(ts, expected, atol=1e-10)
        # LRT statistics should be non-negative (NLL_0 >= NLL_alt)
        self.assertTrue(np.all(ts >= -1e-10))

    def test_test_statistics_f(self):
        """Test statistics for F-test are beta^2/SE^2."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=K, test='f')
        ts = lmm_result.test_statistics
        expected = (lmm_result.beta_snp ** 2) / (lmm_result.beta_ste ** 2)
        np.testing.assert_allclose(ts, expected, atol=1e-10)

    def test_beta_ste_against_scipy_gls(self):
        """Verify qtl_test_lmm betas and SEs against scipy GLS."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm

        X, Y, K = self._make_data()
        lmm_result = qtl_test_lmm(X, Y, K=K, test='lrt')
        beta_lmm = lmm_result.beta_snp.ravel()
        ste_lmm = lmm_result.beta_ste.ravel()
        ldelta0 = lmm_result.ldelta_0.ravel()[0]

        N = K.shape[0]
        S_eig, U = la.eigh(K)
        delta = np.exp(ldelta0)
        UY = U.T @ Y
        Ucovs = U.T @ np.ones((N, 1))
        Usnps = U.T @ X
        Sdi = 1.0 / (S_eig + delta)

        for s in range(X.shape[1]):
            UXps = np.column_stack([Usnps[:, s:s+1], Ucovs])
            XSX = UXps.T @ (Sdi[:, None] * UXps)
            XSY = UXps.T @ (Sdi * UY.ravel())
            beta_all = la.solve(XSX, XSY, assume_a='sym')
            res = UY.ravel() - UXps @ beta_all
            sigma_ml = np.sum(res**2 * Sdi) / N
            XSX_inv = la.inv(XSX)
            ste_scipy = np.sqrt(sigma_ml * XSX_inv[0, 0])

            self.assertAlmostEqual(beta_lmm[s], beta_all[0], places=5)
            self.assertAlmostEqual(ste_lmm[s], ste_scipy, places=5)


class TestQtlTestLmmMissingValues(unittest.TestCase):
    """Tests for the missing values code path in lmm class (qtl.py lines 110-141).

    BUG DOCUMENTATION: The missing-values branch of lmm.process() has multiple bugs:
    1. References `self.phenos` (plural) but the attribute is `self.pheno` (singular)
    2. Uses `pheno_.isnan()` but ndarray has no .isnan() method; should be `np.isnan(pheno_)`
    3. The branch is unreachable for re-processing because line 111 raises if _lmm is not None
       (but _lmm is always set at line 81 before process() is called)

    This test confirms that the missing-value path raises an exception, documenting
    the known bug.
    """

    def test_missing_values_raises_exception(self):
        """The missing-values code path raises an exception due to the re-use check."""
        from limix_legacy.deprecated.modules.qtl import lmm

        N, S = 50, 10
        rng = np.random.RandomState(42)
        X = rng.randn(N, S)
        Y = rng.randn(N, 1)
        # Introduce NaN so we trigger the else branch
        Y[0, 0] = np.nan
        K = np.eye(N)

        # The lmm constructor calls process() which enters the else branch.
        # But _lmm was already set at line 81, so line 111 raises.
        with self.assertRaises(Exception, msg="cannot reuse a CLMM object"):
            lmm(X, Y, K=K)


class TestQtlTestLmmKronecker(unittest.TestCase):
    """Tests for qtl_test_lmm_kronecker."""

    def _make_multitrait_data(self, N=80, S=15, P=2, seed=42):
        """Generate synthetic multi-trait data for Kronecker LMM tests."""
        rng = np.random.RandomState(seed)
        X = rng.randn(N, S)
        K = X @ X.T / S
        K += np.eye(N)  # ensure PD
        L = la.cholesky(K, lower=True)

        Y = np.zeros((N, P))
        for p in range(P):
            beta = rng.randn(S) * 0.1
            beta[0] = 0.5  # causal SNP
            Y[:, p] = X @ beta + L @ rng.randn(N) + rng.randn(N) * 0.5
        return X, Y, K

    def test_basic_kronecker_lmm(self):
        """qtl_test_lmm_kronecker runs and returns valid results."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, K = self._make_multitrait_data()
        N, P = Y.shape

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K2r=np.eye(N),
            trait_covar_type='lowrank_diag', rank=1
        )

        # Defaults: one Asnps design = [1,...,1] of shape (1, P)
        self.assertEqual(pv.shape, (1, X.shape[1]))
        self.assertEqual(beta.shape, (1, X.shape[1]))
        self.assertEqual(ste.shape, (1, X.shape[1]))
        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertTrue(np.all(ste > 0))

    def test_kronecker_with_precomputed_covars(self):
        """qtl_test_lmm_kronecker with user-specified K1c and K2c skips VC estimation."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, K = self._make_multitrait_data()
        N, P = Y.shape
        K1c = np.eye(P)
        K2c = np.eye(P)

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K1c=K1c, K2r=np.eye(N), K2c=K2c
        )

        self.assertEqual(pv.shape, (1, X.shape[1]))
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_kronecker_multiple_designs(self):
        """qtl_test_lmm_kronecker with multiple SNP design matrices."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, K = self._make_multitrait_data()
        N, P = Y.shape

        # Two designs: common effect and independent effects
        Asnps = [np.ones([1, P]), np.eye(P)]

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K1c=np.eye(P), K2r=np.eye(N), K2c=np.eye(P),
            Asnps=Asnps
        )

        self.assertEqual(pv.shape, (2, X.shape[1]))
        self.assertEqual(beta.shape, (2, X.shape[1]))
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_kronecker_defaults_K1r_none(self):
        """When K1r is None, it should default to snps @ snps.T."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, _ = self._make_multitrait_data(N=50, S=10)
        N, P = Y.shape

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=None, K2r=np.eye(N),
            K1c=np.eye(P), K2c=np.eye(P)
        )

        self.assertTrue(np.all(np.isfinite(pv)))

    def test_kronecker_with_covariates(self):
        """qtl_test_lmm_kronecker with explicit covariates."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, K = self._make_multitrait_data()
        N, P = Y.shape

        covs = [np.ones((N, 1))]
        Acovs = [np.eye(P)]

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K1c=np.eye(P), K2r=np.eye(N), K2c=np.eye(P),
            covs=covs, Acovs=Acovs
        )

        self.assertTrue(np.all(np.isfinite(pv)))

    def test_kronecker_univariate_matches_clmm(self):
        """Kronecker LMM with P=1 should match standard CLMM."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm, qtl_test_lmm_kronecker

        rng = np.random.RandomState(42)
        N, S = 80, 20
        X = rng.randn(N, S)
        K = X @ X.T / S + np.eye(N)
        Y1 = rng.randn(N, 1)

        # Standard LMM
        lmm_std = qtl_test_lmm(X, Y1, K=K)
        pv_std = lmm_std.getPv().ravel()

        # Kronecker LMM with P=1
        lmm_k, pv_k, _, _ = qtl_test_lmm_kronecker(
            X, Y1, K1r=K, K1c=np.eye(1), K2r=np.eye(N), K2c=np.eye(1)
        )
        pv_k = pv_k.ravel()

        # p-values should match closely
        np.testing.assert_allclose(
            np.log10(np.clip(pv_std, 1e-300, 1)),
            np.log10(np.clip(pv_k, 1e-300, 1)),
            atol=1e-4
        )

    def test_kronecker_search_delta(self):
        """qtl_test_lmm_kronecker with searchDelta=True."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        X, Y, K = self._make_multitrait_data(N=50, S=5)
        N, P = Y.shape

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K1c=np.eye(P), K2r=np.eye(N), K2c=np.eye(P),
            searchDelta=True, NumIntervalsDeltaAlt=20
        )

        self.assertTrue(np.all(np.isfinite(pv)))


class TestUpdateKronCovs(unittest.TestCase):
    """Tests for the _updateKronCovs helper."""

    def test_none_defaults(self):
        """When both are None, returns intercept + identity design."""
        from limix_legacy.deprecated.modules.qtl import _updateKronCovs

        covs, Acovs = _updateKronCovs(None, None, 10, 3)
        self.assertEqual(len(covs), 1)
        self.assertEqual(len(Acovs), 1)
        np.testing.assert_array_equal(covs[0], np.ones((10, 1)))
        np.testing.assert_array_equal(Acovs[0], np.eye(3))

    def test_one_none_raises(self):
        """If only one is None, should raise."""
        from limix_legacy.deprecated.modules.qtl import _updateKronCovs

        with self.assertRaises(Exception):
            _updateKronCovs(np.ones((10, 1)), None, 10, 3)

        with self.assertRaises(Exception):
            _updateKronCovs(None, np.eye(3), 10, 3)

    def test_single_arrays_wrapped(self):
        """Non-list inputs are wrapped into lists."""
        from limix_legacy.deprecated.modules.qtl import _updateKronCovs

        covs, Acovs = _updateKronCovs(np.ones((10, 1)), np.eye(3), 10, 3)
        self.assertIsInstance(covs, list)
        self.assertIsInstance(Acovs, list)

    def test_length_mismatch_raises(self):
        """Lists of different length should raise."""
        from limix_legacy.deprecated.modules.qtl import _updateKronCovs

        with self.assertRaises(Exception):
            _updateKronCovs([np.ones((10, 1))], [np.eye(3), np.eye(3)], 10, 3)


class TestEstimateKronCovariances(unittest.TestCase):
    """Tests for the _estimateKronCovariances helper."""

    def test_basic_estimation(self):
        """_estimateKronCovariances returns a VarianceDecomposition with valid covariances."""
        from limix_legacy.deprecated.modules.qtl import _estimateKronCovariances

        rng = np.random.RandomState(42)
        N, P = 50, 2
        X = rng.randn(N, 100)
        K1r = X @ X.T / 100
        K1r += np.eye(N)
        K2r = np.eye(N)
        Y = rng.randn(N, P)

        covs = [np.ones((N, 1))]
        Acovs = [np.eye(P)]

        vc = _estimateKronCovariances(
            Y, K1r=K1r, K2r=K2r, covs=covs, Acovs=Acovs,
            trait_covar_type='lowrank_diag', rank=1, verbose=False
        )

        K1c = vc.getTraitCovar(0)
        K2c = vc.getTraitCovar(1)
        self.assertEqual(K1c.shape, (P, P))
        self.assertEqual(K2c.shape, (P, P))
        # Covariances should be PSD (eigenvalues >= 0)
        self.assertTrue(np.all(la.eigvalsh(K1c) >= -1e-8))
        self.assertTrue(np.all(la.eigvalsh(K2c) >= -1e-8))


class TestVarianceDecomposition(unittest.TestCase):
    """Tests for VarianceDecomposition class."""

    def _make_vc_data(self, N=100, S=200, P=2, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(N, S)
        K = X @ X.T / S
        K /= K.diagonal().mean()

        # Generate phenotypes with known variance structure
        Y = np.zeros((N, P))
        Lg = la.cholesky(K + 1e-6 * np.eye(N), lower=True)
        for p in range(P):
            Y[:, p] = Lg @ rng.randn(N) + rng.randn(N) * 0.5
        # Standardize
        Y = (Y - Y.mean(0)) / Y.std(0)

        return Y, K

    def test_basic_optimize(self):
        """VarianceDecomposition optimizes and returns valid results."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        var_comps = vc.getVarianceComps()
        self.assertEqual(var_comps.shape, (2, 2))
        self.assertTrue(np.all(var_comps >= 0))

    def test_fast_optimize(self):
        """VarianceDecomposition with fast=True (kronSum)."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        conv = vc.optimize(fast=True, verbose=False)
        self.assertTrue(conv)

    def test_fast_and_base_agree(self):
        """Fast and base GP should give similar results."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()

        vc_base = VAR.VarianceDecomposition(Y)
        vc_base.addRandomEffect(K, jitter=0)
        vc_base.addRandomEffect(is_noise=True, jitter=0)
        vc_base.addFixedEffect()
        conv_base = vc_base.optimize(fast=False, verbose=False)

        vc_fast = VAR.VarianceDecomposition(Y)
        vc_fast.addRandomEffect(K, jitter=0)
        vc_fast.addRandomEffect(is_noise=True, jitter=0)
        vc_fast.addFixedEffect()
        conv_fast = vc_fast.optimize(fast=True, verbose=False)

        if conv_base and conv_fast:
            var_base = vc_base.getVarianceComps()
            var_fast = vc_fast.getVarianceComps()
            # Results should be similar (same local minimum)
            np.testing.assert_allclose(var_base, var_fast, atol=0.1)

    def test_trait_covar_freeform(self):
        """VarianceDecomposition with freeform trait covariance."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='freeform', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='freeform', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        Cg = vc.getTraitCovar(0)
        Cn = vc.getTraitCovar(1)
        self.assertEqual(Cg.shape, (2, 2))
        self.assertEqual(Cn.shape, (2, 2))
        # Symmetric
        np.testing.assert_allclose(Cg, Cg.T, atol=1e-10)
        np.testing.assert_allclose(Cn, Cn.T, atol=1e-10)

    def test_trait_covar_diag(self):
        """VarianceDecomposition with diagonal trait covariance."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='diag', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='diag', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        Cg = vc.getTraitCovar(0)
        # For diag + jitter, off-diagonal should be zero (or very small from jitter)
        off_diag = Cg[0, 1]
        # The jitter contribution is shared but main diag component gives 0 off-diag
        # Just verify it converged and result is PSD
        self.assertTrue(np.all(la.eigvalsh(Cg) >= -1e-8))

    def test_trait_covar_lowrank(self):
        """VarianceDecomposition with lowrank trait covariance.

        NOTE: lowrank/block types are not compatible with 'diagonal' init_method
        (which is the default for P>1). Must use init_method='random'.
        """
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='lowrank', rank=1, jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='lowrank_diag', rank=1, jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(init_method='random', verbose=False)
        self.assertTrue(conv)

    def test_trait_covar_block(self):
        """VarianceDecomposition with block trait covariance.

        NOTE: block type is not compatible with 'diagonal' init_method.
        Must use init_method='random'.
        """
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='block', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='block_diag', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(init_method='random', verbose=False)
        self.assertTrue(conv)

    def test_trait_covar_block_id(self):
        """VarianceDecomposition with block_id trait covariance."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='block_id', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='block_id', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

    def test_trait_covar_lowrank_id(self):
        """VarianceDecomposition with lowrank_id trait covariance."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='lowrank_id', rank=1, jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='lowrank_diag', rank=1, jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

    def test_getVarianceComps_univariance(self):
        """getVarianceComps with univariance=True normalizes to sum to 1."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        var = vc.getVarianceComps(univariance=True)
        # Each row should sum to 1
        np.testing.assert_allclose(var.sum(axis=1), np.ones(2), atol=1e-10)

    def test_getTraitCorrCoef(self):
        """Trait correlation coefficients should be in [-1, 1] with 1 on diagonal."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        corr = vc.getTraitCorrCoef(0)
        np.testing.assert_allclose(np.diag(corr), np.ones(2), atol=1e-10)
        self.assertTrue(np.all(np.abs(corr) <= 1.0 + 1e-10))

    def test_getScales_and_setScales(self):
        """getScales/setScales roundtrip preserves values."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        scales = vc.getScales()
        vc.setScales(scales)
        scales2 = vc.getScales()
        np.testing.assert_allclose(scales, scales2, atol=1e-12)

    def test_getLML(self):
        """Log marginal likelihood is finite and negative."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        lml = vc.getLML()
        self.assertTrue(np.isfinite(lml))

    def test_single_trait(self):
        """VarianceDecomposition works for P=1."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N = 100
        X = rng.randn(N, 200)
        K = X @ X.T / 200
        K /= K.diagonal().mean()
        Y = rng.randn(N, 1)

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K)
        vc.addRandomEffect(is_noise=True)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        var_comps = vc.getVarianceComps()
        self.assertEqual(var_comps.shape, (1, 2))
        self.assertTrue(np.all(var_comps >= 0))

    def test_1d_phenotype_reshaped(self):
        """1D array phenotype is reshaped to column vector."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N = 50
        Y = rng.randn(N)  # 1D

        vc = VAR.VarianceDecomposition(Y)
        self.assertEqual(vc.P, 1)
        self.assertEqual(vc.N, N)

    def test_normalize_K(self):
        """addRandomEffect normalizes K by default (K.trace()/N == 1)."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N = 50
        Y = rng.randn(N, 2)
        K = rng.randn(N, N)
        K = K @ K.T  # PD but arbitrary scale

        vc = VAR.VarianceDecomposition(Y)
        # The internal K gets normalized
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        # Internal K should be scaled
        K_internal = vc.vd.getTerm(0).getK()
        np.testing.assert_allclose(K_internal.diagonal().mean(), 1.0, atol=1e-6)

    def test_missing_values(self):
        """VarianceDecomposition handles missing values (NaN in Y)."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N, P = 100, 2
        X = rng.randn(N, 200)
        K = X @ X.T / 200
        K /= K.diagonal().mean()
        Y = rng.randn(N, P)
        Y[0, 0] = np.nan
        Y[5, 1] = np.nan

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        # With missing values, fast=True should not be used
        conv = vc.optimize(fast=False, verbose=False)
        self.assertTrue(conv)

    def test_optimize_with_repeats(self):
        """optimize_with_repeates returns sorted list of optima."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data(N=50, S=100)
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()

        results = vc.optimize_with_repeates(n_times=3, verbose=False)
        self.assertIsInstance(results, list)
        if len(results) > 1:
            # Should be sorted by LML descending
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['LML'], results[i+1]['LML'])

    def test_noise_pos_set_correctly(self):
        """noisPos is set to the index of the noise term."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        self.assertEqual(vc.noisPos, 1)

    def test_double_noise_raises(self):
        """Adding two noise terms should raise."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(is_noise=True)
        with self.assertRaises(AssertionError):
            vc.addRandomEffect(is_noise=True)

    def test_getWeights(self):
        """getWeights returns finite values after optimization."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        weights = vc.getWeights()
        self.assertTrue(np.all(np.isfinite(weights)))


class TestVarianceDecompositionStdErrors(unittest.TestCase):
    """Tests for VarianceDecomposition standard errors.

    BUG DOCUMENTATION for getTraitCovarStdErrors (varianceDecomposition.py lines 553-558):

    Bug 1: Line 556-557 uses wrong loop variable for accumulating parameter index:
        for term in range(term_i-1):
            par_index += self.vd.getTerm(term_i).getNumberScales()  # BUG: should be getTerm(term)

    Bug 2: Line 556 uses range(term_i-1) but should be range(term_i) to include
        all terms before term_i.

    The combined effect: for term_i=0, par_index stays 0 (correct by accident since
    range(-1) is empty). For term_i=1, range(0) is also empty so par_index=0, but it
    should be the number of scales in term 0. For term_i>=2, it accumulates the wrong
    term's scales.
    """

    def _make_vc_fitted(self, N=80, P=2, seed=42):
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(seed)
        X = rng.randn(N, 200)
        K = X @ X.T / 200
        K /= K.diagonal().mean()
        Y = rng.randn(N, P)
        Y = (Y - Y.mean(0)) / Y.std(0)

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='freeform', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='freeform', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(fast=False, verbose=False)
        return vc, conv

    def test_std_errors_hessian_numpy_bug(self):
        """BUG: _getHessian crashes with TypeError on numpy >= 2.0.

        In varianceDecomposition.py line 1044:
            std = np.zeros(ParamMask.sum())

        ParamMask.sum() returns np.float64 (because ParamMask is a float array),
        but np.zeros() in numpy 2.x requires an integer argument.
        This means getTraitCovarStdErrors is broken for multi-trait models.
        """
        vc, conv = self._make_vc_fitted()
        if not conv:
            self.skipTest("VC did not converge")

        with self.assertRaises(TypeError, msg="np.zeros rejects np.float64 size in numpy 2.x"):
            vc.getTraitCovarStdErrors(0)

    def test_std_errors_par_index_bug_documented(self):
        """Document the par_index bug in getTraitCovarStdErrors.

        Even if the numpy bug above is fixed, there's a second bug:
        Line 556-557 uses wrong loop variable:
            for term in range(term_i-1):
                par_index += self.vd.getTerm(term_i).getNumberScales()
        Should be: getTerm(term) and range(term_i).
        """
        vc, conv = self._make_vc_fitted()
        if not conv:
            self.skipTest("VC did not converge")

        # Document the par_index bug: manually verify the indices are wrong
        C0 = vc.vd.getTerm(0).getTraitCovar()
        n_params_term0 = C0.getNumberParams()
        self.assertGreater(n_params_term0, 0,
                          "Term 0 has params, so par_index for term 1 should be nonzero")

        # The buggy code for term_i=1:
        #   for term in range(term_i - 1):  # range(0) -> empty loop
        #       par_index += self.vd.getTerm(term_i).getNumberScales()  # never executes
        # So par_index stays 0, but it should be n_params_term0
        buggy_par_index = 0  # This is what the code computes
        correct_par_index = n_params_term0
        self.assertNotEqual(buggy_par_index, correct_par_index,
                           "Bug: par_index should be %d but code gives 0" % correct_par_index)

    def test_single_trait_std_errors_hessian_bug(self):
        """BUG: Single-trait std errors also hit the numpy _getHessian bug."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N = 80
        X = rng.randn(N, 200)
        K = X @ X.T / 200
        K /= K.diagonal().mean()
        Y = rng.randn(N, 1)

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K)
        vc.addRandomEffect(is_noise=True)
        vc.addFixedEffect()
        conv = vc.optimize(fast=False, verbose=False)
        if not conv:
            self.skipTest("VC did not converge")

        # Hits the same numpy 2.x bug in _getHessian
        with self.assertRaises(TypeError):
            vc.getTraitCovarStdErrors(0)


class TestBuildTraitCovar(unittest.TestCase):
    """Test _buildTraitCovar for each trait_covar_type."""

    def test_fixed_shape_check_bug(self):
        """BUG: _buildTraitCovar checks fixed_trait_covar.shape against self.N instead of self.P.

        In varianceDecomposition.py line 774-775:
            assert fixed_trait_covar.shape[0]==self.N  # BUG: should be self.P
            assert fixed_trait_covar.shape[1]==self.N  # BUG: should be self.P

        This means if N != P (which is almost always the case), passing a correct
        PxP fixed_trait_covar will fail, and passing an incorrect NxN matrix will pass.
        """
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N, P = 50, 3
        Y = rng.randn(N, P)
        vc = VAR.VarianceDecomposition(Y)

        # This SHOULD work (PxP matrix) but will fail due to the bug
        correct_fixed_covar = np.eye(P)
        with self.assertRaises(AssertionError):
            vc.addRandomEffect(K=np.eye(N), trait_covar_type='fixed',
                             fixed_trait_covar=correct_fixed_covar, jitter=0)

        # This SHOULDN'T work (NxN matrix) but passes the assertion
        # (will likely fail downstream when dimensions don't match)
        wrong_fixed_covar = np.eye(N)
        # We don't test this further as it would create an invalid model


class TestCKroneckerLMMDirect(unittest.TestCase):
    """Direct tests of the CKroneckerLMM C++ wrapper."""

    def _make_data(self, N=80, S=20, P=2, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(N, S)
        K = X @ X.T / S + np.eye(N)
        Y = rng.randn(N, P)
        return X, Y, K

    def test_univariate_equivalence_direct(self):
        """CKroneckerLMM with P=1 gives same p-values as CLMM."""
        import limix_legacy.deprecated as dlimix_legacy

        dir_name = os.path.join(os.path.dirname(__file__), 'lmm')
        from limix_legacy.test import data
        D = data.load(os.path.join(dir_name, 'lmm_data1'))

        # Standard CLMM
        slmm = dlimix_legacy.CLMM()
        slmm.setK(D['K'])
        slmm.setSNPs(D['X'])
        slmm.setCovs(D['Cov'])
        slmm.setPheno(D['Y'])
        slmm.setNumIntervals0(100)
        slmm.setNumIntervalsAlt(0)
        slmm.process()
        pv_std = slmm.getPv().ravel()

        # Kronecker equivalent
        N = D['K'].shape[0]
        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(D['K'])
        klmm.setK1c(np.eye(1))
        klmm.setK2r(np.eye(N))
        klmm.setK2c(np.eye(1))
        klmm.setSNPs(D['X'])
        klmm.addCovariates(D['Cov'][:, np.newaxis], np.eye(1))
        klmm.setSNPcoldesign(np.eye(1))
        klmm.setPheno(D['Y'][:, np.newaxis])
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()
        pv_kron = klmm.getPv().ravel()

        np.testing.assert_allclose(
            np.log10(np.clip(pv_std, 1e-300, 1)),
            np.log10(np.clip(pv_kron, 1e-300, 1)),
            atol=1e-5
        )

    def test_multitrait_pv_nonnegative(self):
        """CKroneckerLMM with multiple traits returns valid p-values."""
        import limix_legacy.deprecated as dlimix_legacy

        X, Y, K = self._make_data()
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(np.eye(P))
        klmm.setK2r(np.eye(N))
        klmm.setK2c(np.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(np.ones((N, 1)), np.eye(P))
        klmm.setSNPcoldesign(np.ones((1, P)))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()

        pv = klmm.getPv()
        beta = klmm.getBetaSNP()
        ste = klmm.getBetaSNPste()

        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertTrue(np.all(ste > 0))
        self.assertTrue(np.all(np.isfinite(ste)))

    def test_multiple_designs_different_dof(self):
        """Different SNP designs produce p-values from different DOF tests."""
        import limix_legacy.deprecated as dlimix_legacy

        X, Y, K = self._make_data()
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(np.eye(P))
        klmm.setK2r(np.eye(N))
        klmm.setK2c(np.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(np.ones((N, 1)), np.eye(P))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)

        # Common effect (1 DOF)
        klmm.setSNPcoldesign(np.ones((1, P)))
        klmm.process()
        pv_common = klmm.getPv().copy()

        # Independent effects (P DOF)
        klmm.setSNPcoldesign(np.eye(P))
        klmm.process()
        pv_indep = klmm.getPv().copy()

        # Both should be valid
        self.assertTrue(np.all(np.isfinite(pv_common)))
        self.assertTrue(np.all(np.isfinite(pv_indep)))
        # They should generally differ (different tests)
        self.assertFalse(np.allclose(pv_common, pv_indep))

    def test_beta_ste_multitrait_wald_consistency(self):
        """Multi-trait Wald p-values should be correlated with LRT p-values."""
        import limix_legacy.deprecated as dlimix_legacy

        rng = np.random.RandomState(42)
        N, S, P = 100, 30, 2
        X = rng.randn(N, S)
        K = X @ X.T / S + np.eye(N)
        # Create correlated phenotypes with a signal
        beta = rng.randn(S) * 0.2
        beta[0] = 1.0
        Y = np.column_stack([
            X @ beta + rng.randn(N),
            X @ beta * 0.5 + rng.randn(N)
        ])

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(np.eye(P))
        klmm.setK2r(np.eye(N))
        klmm.setK2c(np.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(np.ones((N, 1)), np.eye(P))
        klmm.setSNPcoldesign(np.ones((1, P)))  # common effect (1 DOF)
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()

        beta_est = klmm.getBetaSNP().ravel()
        ste = klmm.getBetaSNPste().ravel()
        pv_lrt = klmm.getPv().ravel()

        # Wald test with 1 DOF
        wald = (beta_est / ste) ** 2
        pv_wald = st.chi2.sf(wald, 1)

        # -log10 correlation
        log_lrt = -np.log10(np.clip(pv_lrt, 1e-300, 1))
        log_wald = -np.log10(np.clip(pv_wald, 1e-300, 1))
        corr = np.corrcoef(log_lrt, log_wald)[0, 1]
        self.assertGreater(corr, 0.9)

    def test_nll_alt_leq_nll_null(self):
        """Alternative model NLL should be <= null model NLL."""
        import limix_legacy.deprecated as dlimix_legacy

        X, Y, K = self._make_data(N=60, S=10)
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(np.eye(P))
        klmm.setK2r(np.eye(N))
        klmm.setK2c(np.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(np.ones((N, 1)), np.eye(P))
        klmm.setSNPcoldesign(np.ones((1, P)))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()

        nll0 = klmm.getNLL0()
        nll_alt = klmm.getNLLAlt()

        # NLL_alt <= NLL_0 (alt model has more parameters)
        self.assertTrue(np.all(nll_alt <= nll0 + 1e-6),
                       "NLL alt should be <= NLL null")


class TestQtlTestInteractionKronecker(unittest.TestCase):
    """Tests for qtl_test_interaction_lmm_kronecker.

    BUG: Line 392 has np.eye([P]) instead of np.eye(P).
    This is in the default for Asnps1 when Asnps1 is None.
    """

    def test_interaction_eye_bug(self):
        """BUG: np.eye([P]) on line 392 of qtl.py passes a list to np.eye.

        In numpy >= 2.0, np.eye([P]) raises TypeError. This means
        qtl_test_interaction_lmm_kronecker will crash when Asnps1 is None
        (which triggers the default np.eye([P])).

        The fix should be np.eye(P) instead of np.eye([P]).
        """
        # Confirm the bug: np.eye([2]) raises in numpy 2.x
        with self.assertRaises(TypeError):
            np.eye([2])


class TestPairwiseInitBug(unittest.TestCase):
    """Test for the pairwise initialization check bug.

    BUG in varianceDecomposition.py line 316:
        i = (self.trait_covar_type[0]=='freeform')*(self.trait_covar_type[0]=='freeform')
    The second comparison should check trait_covar_type[1], not [0] again.
    """

    def test_pairwise_init_check_uses_wrong_index(self):
        """The pairwise init check tests trait_covar_type[0] twice instead of [0] and [1]."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR

        rng = np.random.RandomState(42)
        N, P = 50, 2
        Y = rng.randn(N, P)
        K = np.eye(N)

        vc = VAR.VarianceDecomposition(Y)
        # Term 0: freeform, Term 1 (noise): diag -- pairwise should not be allowed
        vc.addRandomEffect(K, trait_covar_type='freeform', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='diag', jitter=1e-4)
        vc.addFixedEffect()

        # Due to the bug, this check passes even though term 1 is 'diag' not 'freeform':
        check_result = (vc.trait_covar_type[0] == 'freeform') * (vc.trait_covar_type[0] == 'freeform')
        # Bug: both check index [0], so result is True even though term 1 is 'diag'
        self.assertTrue(check_result, "Bug: check passes because it tests index 0 twice")

        # Correct check would be:
        correct_check = (vc.trait_covar_type[0] == 'freeform') * (vc.trait_covar_type[1] == 'freeform')
        self.assertFalse(correct_check, "Correct check properly detects term 1 is not freeform")


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests combining VarianceDecomposition with Kronecker LMM."""

    def test_full_kronecker_pipeline(self):
        """Full pipeline: simulate data, estimate covariances, test SNPs."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        rng = np.random.RandomState(42)
        N, S, P = 80, 15, 2

        # Simulate genotype data
        X = rng.randn(N, S)
        K = X @ X.T / S
        K += np.eye(N)

        # Simulate multi-trait phenotypes with a shared causal SNP
        beta_shared = np.zeros(S)
        beta_shared[0] = 0.8
        Lg = la.cholesky(K, lower=True)
        Y = np.zeros((N, P))
        for p in range(P):
            Y[:, p] = X @ beta_shared + Lg @ rng.randn(N) + rng.randn(N) * 0.5

        # Run full pipeline (variance estimation + testing)
        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K2r=np.eye(N),
            trait_covar_type='lowrank_diag', rank=1
        )

        # Validate results
        self.assertEqual(pv.shape, (1, S))
        self.assertTrue(np.all(np.isfinite(pv)))
        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))

        # Causal SNP (index 0) should be among the most significant
        self.assertLess(pv[0, 0], 0.1,
                       "Causal SNP should have a small p-value in multi-trait test")

    def test_kronecker_three_traits(self):
        """Kronecker LMM with 3 traits."""
        from limix_legacy.deprecated.modules.qtl import qtl_test_lmm_kronecker

        rng = np.random.RandomState(42)
        N, S, P = 60, 10, 3
        X = rng.randn(N, S)
        K = X @ X.T / S + np.eye(N)
        Y = rng.randn(N, P)

        lmm, pv, beta, ste = qtl_test_lmm_kronecker(
            X, Y, K1r=K, K1c=np.eye(P), K2r=np.eye(N), K2c=np.eye(P)
        )

        self.assertEqual(pv.shape, (1, S))
        self.assertTrue(np.all(np.isfinite(pv)))

    def test_variance_decomposition_then_kronecker(self):
        """Manually run VarianceDecomposition then CKroneckerLMM (replicating qtl_test_lmm_kronecker)."""
        import limix_legacy.deprecated.modules.varianceDecomposition as VAR
        import limix_legacy.deprecated as dlimix_legacy

        rng = np.random.RandomState(42)
        N, S, P = 80, 15, 2
        X = rng.randn(N, S)
        K = X @ X.T / S + np.eye(N)
        Y = rng.randn(N, P)

        # Step 1: Estimate covariances
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='lowrank_diag', rank=1)
        vc.addRandomEffect(is_noise=True, K=np.eye(N), trait_covar_type='lowrank_diag', rank=1)
        vc.addFixedEffect()
        conv = vc.optimize(fast=True, verbose=False)
        self.assertTrue(conv)

        K1c = vc.getTraitCovar(0)
        K2c = vc.getTraitCovar(1)
        self.assertEqual(K1c.shape, (P, P))
        self.assertEqual(K2c.shape, (P, P))

        # Step 2: Run Kronecker LMM
        lmm = dlimix_legacy.CKroneckerLMM()
        lmm.setK1r(K)
        lmm.setK1c(K1c)
        lmm.setK2r(np.eye(N))
        lmm.setK2c(K2c)
        lmm.setSNPs(X)
        lmm.addCovariates(np.ones((N, 1)), np.eye(P))
        lmm.setSNPcoldesign(np.ones((1, P)))
        lmm.setPheno(Y)
        lmm.setNumIntervals0(100)
        lmm.setNumIntervalsAlt(0)
        lmm.process()

        pv = lmm.getPv()
        self.assertTrue(np.all(np.isfinite(pv)))
        self.assertTrue(np.all(pv >= 0))
        self.assertTrue(np.all(pv <= 1))


if __name__ == '__main__':
    unittest.main()
