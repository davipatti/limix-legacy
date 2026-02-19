"""LMM testing code"""
import unittest
import numpy as NP
import pdb
import limix_legacy
import limix_legacy.deprecated as dlimix_legacy
from limix_legacy.test import data
import os


class CKroneckerLMM_test(unittest.TestCase):
    """test class for CLMM"""

    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(os.path.realpath(__file__))

    def test_lmm(self):
        """basic test, comparing pv to a standard LMM equivalent"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #construct Kronecker LMM model which has the special case of standard LMM
            #covar1: genotype matrix
            K1r = D['K']
            K1c = NP.eye(1)
            K2r = NP.eye(D['K'].shape[0])
            K2c = NP.eye(1)
            A   = NP.eye(1)
            Acov = NP.eye(1)
            Xcov  = D['Cov'][:,NP.newaxis]
            X      = D['X']
            Y      = D['Y'][:,NP.newaxis]

            lmm = dlimix_legacy.CKroneckerLMM()
            lmm.setK1r(K1r)
            lmm.setK1c(K1c)
            lmm.setK2r(K2r)
            lmm.setK2c(K2c)

            lmm.setSNPs(X)
            #add covariates
            lmm.addCovariates(Xcov,Acov)
            #add SNP design
            lmm.setSNPcoldesign(A)
            lmm.setPheno(Y)
            lmm.setNumIntervalsAlt(0)
            lmm.setNumIntervals0(100)

            lmm.process()
            pv = lmm.getPv().ravel()
            D2= ((NP.log10(pv)-NP.log10(D['pv']))**2)
            RV = NP.sqrt(D2.mean())
            #print "\n"
            #print pv[0:10]
            #print D['pv'][0:10]
            #print RV
            #pdb.set_trace()
            self.assertTrue(RV<1E-6)

    def test_lmm2(self):
        """another test, establishing an lmm-equivalent by a design matrix choice"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #construct Kronecker LMM model which has the special case of standard LMM
            #covar1: genotype matrix
            N = D['K'].shape[0]
            P = 3
            K1r = D['K']
            #K1c = NP.zeros([2,2])
            #K1c[0,0] = 1
            K1c = NP.eye(P)
            K2r = NP.eye(N)
            K2c = NP.eye(P)

            #A   = NP.zeros([1,2])
            #A[0,0] =1
            A = NP.eye(P)
            Acov = NP.eye(P)
            Xcov = D['Cov'][:,NP.newaxis]
            X      = D['X']
            Y      = D['Y'][:,NP.newaxis]
            Y      = NP.tile(Y,(1,P))

            lmm = dlimix_legacy.CKroneckerLMM()
            lmm.setK1r(K1r)
            lmm.setK1c(K1c)
            lmm.setK2r(K2r)
            lmm.setK2c(K2c)

            lmm.setSNPs(X)
            #add covariates
            lmm.addCovariates(Xcov,Acov)
            #add SNP design
            lmm.setSNPcoldesign(A)
            lmm.setPheno(Y)
            lmm.setNumIntervalsAlt(0)
            lmm.setNumIntervals0(100)

            lmm.process()

            #get p-values with P-dof:
            pv_Pdof = lmm.getPv().ravel()
            #transform in P-values with a single DOF:
            import scipy.stats as st
            lrt = st.chi2.isf(pv_Pdof,P)/P
            pv = st.chi2.sf(lrt,1)
            #compare with single DOF P-values:
            D2= ((NP.log10(pv)-NP.log10(D['pv']))**2)
            RV = NP.sqrt(D2.mean())
            #print "\n"
            #print pv[0:10]
            #print D['pv'][0:10]
            #print RV
            #pdb.set_trace()
            self.assertTrue(RV<1E-6)



    def test_permutation(self):
        #test permutation function
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            perm = NP.random.permutation(D['X'].shape[0])
            #1. set permuattion
            lmm = dlimix_legacy.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            if 1:
                #pdb.set_trace()
                perm = NP.array(perm,dtype='int32')#Windows needs int32 as long -> fix interface to accept int64 types
            lmm.setPermutation(perm)
            lmm.process()
            pv_perm1 = lmm.getPv().ravel()
            #2. do by hand
            lmm = dlimix_legacy.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'][perm])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv_perm2 = lmm.getPv().ravel()
            D2 = (NP.log10(pv_perm1)-NP.log10(pv_perm2))**2
            RV = NP.sqrt(D2.mean())
            self.assertTrue(RV<1E-6)



    def test_beta_ste(self):
        """test Kronecker LMM standard errors via Wald-LRT consistency in the univariate case.

        Validates SE by checking that Wald p-values (beta^2/SE^2) closely match
        LRT p-values. Also verifies betas match the standard CLMM.
        """
        import scipy.stats as st
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            K1r = D['K']
            K1c = NP.eye(1)
            K2r = NP.eye(D['K'].shape[0])
            K2c = NP.eye(1)
            A   = NP.eye(1)
            Acov = NP.eye(1)
            Xcov  = D['Cov'][:,NP.newaxis]
            X      = D['X']
            Y      = D['Y'][:,NP.newaxis]

            #1. Kronecker LMM
            klmm = dlimix_legacy.CKroneckerLMM()
            klmm.setK1r(K1r)
            klmm.setK1c(K1c)
            klmm.setK2r(K2r)
            klmm.setK2c(K2c)
            klmm.setSNPs(X)
            klmm.addCovariates(Xcov,Acov)
            klmm.setSNPcoldesign(A)
            klmm.setPheno(Y)
            klmm.setNumIntervalsAlt(0)
            klmm.setNumIntervals0(100)
            klmm.process()
            beta_kron = klmm.getBetaSNP().ravel()
            ste_kron  = klmm.getBetaSNPste().ravel()
            pv_lrt    = klmm.getPv().ravel()

            #2. Standard CLMM (for beta comparison only)
            slmm = dlimix_legacy.CLMM()
            slmm.setK(D['K'])
            slmm.setSNPs(X)
            slmm.setCovs(D['Cov'])
            slmm.setPheno(D['Y'])
            slmm.setNumIntervals0(100)
            slmm.setNumIntervalsAlt(0)
            slmm.process()
            beta_std = slmm.getBetaSNP().ravel()

            #3. Compare betas (should match exactly)
            D2_beta = ((beta_kron - beta_std)**2)
            RV_beta = NP.sqrt(D2_beta.mean())
            self.assertTrue(RV_beta < 1E-6, "beta mismatch: RMSE=%.2e" % RV_beta)

            #4. Sanity: SE should be positive and finite
            self.assertTrue(NP.all(ste_kron > 0), "SE should be positive")
            self.assertTrue(NP.all(NP.isfinite(ste_kron)), "SE should be finite")

            #5. Wald p-values from SE should match LRT p-values closely
            wald = (beta_kron / ste_kron)**2
            pv_wald = st.chi2.sf(wald, 1)
            # Correlation in -log10 space
            log_lrt = -NP.log10(NP.clip(pv_lrt, 1e-300, 1))
            log_wald = -NP.log10(NP.clip(pv_wald, 1e-300, 1))
            corr = NP.corrcoef(log_lrt, log_wald)[0,1]
            self.assertTrue(corr > 0.999, "Wald-LRT correlation too low: r=%.6f" % corr)

    def test_beta_ste_multitrait(self):
        """test SE in the multi-trait case: SE > 0, finite, and Wald test ~ LRT p-values"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            N = D['K'].shape[0]
            P = 2
            K1r = D['K']
            K1c = NP.eye(P)
            K2r = NP.eye(N)
            K2c = NP.eye(P)
            # Use common-effect design (1 DOF)
            A = NP.ones([1,P])
            Acov = NP.eye(P)
            Xcov = D['Cov'][:,NP.newaxis]
            X = D['X']
            Y = NP.column_stack([D['Y'], D['Y'] + NP.random.RandomState(0).randn(N)*0.1])

            klmm = dlimix_legacy.CKroneckerLMM()
            klmm.setK1r(K1r)
            klmm.setK1c(K1c)
            klmm.setK2r(K2r)
            klmm.setK2c(K2c)
            klmm.setSNPs(X)
            klmm.addCovariates(Xcov,Acov)
            klmm.setSNPcoldesign(A)
            klmm.setPheno(Y)
            klmm.setNumIntervalsAlt(0)
            klmm.setNumIntervals0(100)
            klmm.process()

            beta = klmm.getBetaSNP().ravel()
            ste  = klmm.getBetaSNPste().ravel()

            # Sanity checks
            self.assertTrue(NP.all(ste > 0), "SE should be positive")
            self.assertTrue(NP.all(NP.isfinite(ste)), "SE should be finite")

            # Wald statistic should be roughly consistent with LRT
            import scipy.stats as st
            pv_lrt = klmm.getPv().ravel()
            wald = (beta / ste)**2
            pv_wald = st.chi2.sf(wald, 1)
            # Correlation in -log10 space should be high
            log_lrt = -NP.log10(NP.clip(pv_lrt, 1e-300, 1))
            log_wald = -NP.log10(NP.clip(pv_wald, 1e-300, 1))
            corr = NP.corrcoef(log_lrt, log_wald)[0,1]
            self.assertTrue(corr > 0.9, "Wald and LRT p-values should be correlated (r=%.3f)" % corr)


class CKroneckerLMM_synth_test(unittest.TestCase):
    """Tests of CKroneckerLMM using synthetic data (no reference files)."""

    def _make_data(self, N=80, S=20, P=2, seed=42):
        rng = NP.random.RandomState(seed)
        X = rng.randn(N, S)
        K = X @ X.T / S + NP.eye(N)
        Y = rng.randn(N, P)
        return X, Y, K

    def test_multitrait_pv_nonnegative(self):
        """CKroneckerLMM with multiple traits returns valid p-values."""
        X, Y, K = self._make_data()
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(NP.eye(P))
        klmm.setK2r(NP.eye(N))
        klmm.setK2c(NP.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(NP.ones((N, 1)), NP.eye(P))
        klmm.setSNPcoldesign(NP.ones((1, P)))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()

        pv = klmm.getPv()
        beta = klmm.getBetaSNP()
        ste = klmm.getBetaSNPste()

        self.assertTrue(NP.all(pv >= 0))
        self.assertTrue(NP.all(pv <= 1))
        self.assertTrue(NP.all(NP.isfinite(beta)))
        self.assertTrue(NP.all(ste > 0))
        self.assertTrue(NP.all(NP.isfinite(ste)))

    def test_multiple_designs_different_dof(self):
        """Different SNP designs produce p-values from different DOF tests."""
        X, Y, K = self._make_data()
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(NP.eye(P))
        klmm.setK2r(NP.eye(N))
        klmm.setK2c(NP.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(NP.ones((N, 1)), NP.eye(P))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)

        # Common effect (1 DOF)
        klmm.setSNPcoldesign(NP.ones((1, P)))
        klmm.process()
        pv_common = klmm.getPv().copy()

        # Independent effects (P DOF)
        klmm.setSNPcoldesign(NP.eye(P))
        klmm.process()
        pv_indep = klmm.getPv().copy()

        # Both should be valid
        self.assertTrue(NP.all(NP.isfinite(pv_common)))
        self.assertTrue(NP.all(NP.isfinite(pv_indep)))
        # They should generally differ (different tests)
        self.assertFalse(NP.allclose(pv_common, pv_indep))

    def test_beta_ste_multitrait_wald_consistency(self):
        """Multi-trait Wald p-values should be correlated with LRT p-values."""
        import scipy.stats as st

        rng = NP.random.RandomState(42)
        N, S, P = 100, 30, 2
        X = rng.randn(N, S)
        K = X @ X.T / S + NP.eye(N)
        # Create correlated phenotypes with a signal
        beta = rng.randn(S) * 0.2
        beta[0] = 1.0
        Y = NP.column_stack([
            X @ beta + rng.randn(N),
            X @ beta * 0.5 + rng.randn(N)
        ])

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(NP.eye(P))
        klmm.setK2r(NP.eye(N))
        klmm.setK2c(NP.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(NP.ones((N, 1)), NP.eye(P))
        klmm.setSNPcoldesign(NP.ones((1, P)))  # common effect (1 DOF)
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
        log_lrt = -NP.log10(NP.clip(pv_lrt, 1e-300, 1))
        log_wald = -NP.log10(NP.clip(pv_wald, 1e-300, 1))
        corr = NP.corrcoef(log_lrt, log_wald)[0, 1]
        self.assertGreater(corr, 0.9)

    def test_nll_alt_leq_nll_null(self):
        """Alternative model NLL should be <= null model NLL."""
        X, Y, K = self._make_data(N=60, S=10)
        N, P = Y.shape

        klmm = dlimix_legacy.CKroneckerLMM()
        klmm.setK1r(K)
        klmm.setK1c(NP.eye(P))
        klmm.setK2r(NP.eye(N))
        klmm.setK2c(NP.eye(P))
        klmm.setSNPs(X)
        klmm.addCovariates(NP.ones((N, 1)), NP.eye(P))
        klmm.setSNPcoldesign(NP.ones((1, P)))
        klmm.setPheno(Y)
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()

        nll0 = klmm.getNLL0()
        nll_alt = klmm.getNLLAlt()

        # NLL_alt <= NLL_0 (alt model has more parameters)
        self.assertTrue(NP.all(nll_alt <= nll0 + 1e-6),
                       "NLL alt should be <= NLL null")

    def test_univariate_equivalence_direct(self):
        """CKroneckerLMM with P=1 gives same p-values as CLMM."""
        dir_name = os.path.dirname(os.path.realpath(__file__))
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
        klmm.setK1c(NP.eye(1))
        klmm.setK2r(NP.eye(N))
        klmm.setK2c(NP.eye(1))
        klmm.setSNPs(D['X'])
        klmm.addCovariates(D['Cov'][:, NP.newaxis], NP.eye(1))
        klmm.setSNPcoldesign(NP.eye(1))
        klmm.setPheno(D['Y'][:, NP.newaxis])
        klmm.setNumIntervals0(100)
        klmm.setNumIntervalsAlt(0)
        klmm.process()
        pv_kron = klmm.getPv().ravel()

        NP.testing.assert_allclose(
            NP.log10(NP.clip(pv_std, 1e-300, 1)),
            NP.log10(NP.clip(pv_kron, 1e-300, 1)),
            atol=1e-5
        )


class CInteractLMM_test:
    """Interaction test"""
    def __init__(self):
        pass

    def test_all(self):
        RV = False
        #print 'CInteractLMM IMPLEMENTED %s' % message(RV)


if __name__ == '__main__':
    unittest.main()
