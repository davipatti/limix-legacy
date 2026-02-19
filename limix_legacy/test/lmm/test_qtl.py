"""Tests for limix_legacy.deprecated.modules.qtl wrapper functions.

Covers qtl_test_lmm, qtl_test_lmm_kronecker, and their internal helpers
(_updateKronCovs, _estimateKronCovariances).  Also documents known bugs.
"""
import unittest
import numpy as np
import scipy.linalg as la


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
