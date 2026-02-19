"""LMM testing code"""
import unittest
import numpy as NP
import pdb
import limix_legacy
import limix_legacy.deprecated as dlimix_legacy
from limix_legacy.test import data
import os
import sys


class CLMM_test(unittest.TestCase):
    """test class for CLMM"""

    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(os.path.realpath(__file__))

    def test_lmm1(self):
        """basic test, comapring pv"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            lmm = dlimix_legacy.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv = lmm.getPv().ravel()
            BetaSte = lmm.getBetaSNPste().ravel()
            Beta = lmm.getBetaSNP()
            D2pv= ((NP.log10(pv)-NP.log10(D['pv']))**2)
            # D2Beta= (Beta-D['Beta'])**2
            # D2BetaSte = (BetaSte-D['BetaSte'])**2
            RV = NP.sqrt(D2pv.mean())<1E-6
            # RV = RV & (D2Beta.mean()<1E-6)
            # RV = RV & (D2BetaSte.mean()<1E-6)
            self.assertTrue(RV)

    def test_exceptions(self):
        D = data.load(os.path.join(self.dir_name,self.datasets[0]))
        lmm  = dlimix_legacy.CLMM()
        N = 100
        K = NP.eye(N)
        X = NP.random.randn(N,100)
        Y = NP.random.randn(N+1,1)
        Cov = NP.random.randn(N,1)
        lmm.setK(K)
        lmm.setSNPs(X)
        lmm.setCovs(Cov)
        lmm.setPheno(Y)
        try:
            lmm.process()
        except Exception as e:
            self.assertTrue(1==1)
            pass

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



    def test_beta_ste_against_scipy(self):
        """Test CLMM standard errors against scipy GLS ground truth.

        For each SNP, we:
        1. Eigendecompose K, rotate data into the eigenbasis
        2. Form the GLS weight matrix S_inv = diag(1/(S + delta))
        3. Solve the weighted least squares problem with scipy
        4. Compute SE = sqrt(sigma_ml * diag((X'S_inv X)^{-1}))
        5. Compare against CLMM's getBetaSNPste()
        """
        import scipy.linalg as la

        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name, dn))
            K = D['K']
            X_snps = D['X']
            covs = D['Cov']
            Y = D['Y']
            N = K.shape[0]
            num_snps = X_snps.shape[1]

            # --- Run CLMM ---
            lmm = dlimix_legacy.CLMM()
            lmm.setK(K)
            lmm.setSNPs(X_snps)
            lmm.setCovs(covs)
            lmm.setPheno(Y)
            lmm.setNumIntervals0(100)
            lmm.setNumIntervalsAlt(0)
            lmm.process()
            beta_clmm = lmm.getBetaSNP().ravel()
            ste_clmm = lmm.getBetaSNPste().ravel()
            ldelta0 = lmm.getLdelta0().ravel()[0]

            # --- Scipy ground truth ---
            # Eigendecompose K (same transform CLMM uses internally)
            S_eig, U = la.eigh(K)
            delta = NP.exp(ldelta0)
            # Rotate data into eigenbasis
            UY = U.T @ Y
            Ucovs = U.T @ covs
            Usnps = U.T @ X_snps
            # GLS weights: 1/(S + delta)
            Sdi = 1.0 / (S_eig + delta)

            ste_scipy = NP.zeros(num_snps)
            beta_scipy = NP.zeros(num_snps)
            for s in range(num_snps):
                # Design matrix: [SNP | covariates] — same column order as CLMM
                UXps = NP.column_stack([Usnps[:, s:s+1], Ucovs])
                # Weighted normal equations: X' diag(Sdi) X
                XSX = UXps.T @ (Sdi[:, None] * UXps)
                XSY = UXps.T @ (Sdi * UY.ravel())
                # Solve for beta
                beta_all = la.solve(XSX, XSY, assume_a='sym')
                beta_scipy[s] = beta_all[0]  # SNP is column 0
                # Residuals and ML sigma
                res = UY.ravel() - UXps @ beta_all
                sigma_ml = NP.sum(res**2 * Sdi) / N
                # SE from inverse of information matrix
                XSX_inv = la.inv(XSX)
                ste_scipy[s] = NP.sqrt(sigma_ml * XSX_inv[0, 0])

            # Betas should match
            rmse_beta = NP.sqrt(NP.mean((beta_clmm - beta_scipy)**2))
            self.assertTrue(rmse_beta < 1e-6,
                            "beta mismatch: RMSE=%.2e" % rmse_beta)

            # Standard errors should match
            rmse_ste = NP.sqrt(NP.mean((ste_clmm - ste_scipy)**2))
            self.assertTrue(rmse_ste < 1e-6,
                            "SE mismatch vs scipy: RMSE=%.2e" % rmse_ste)


class CInteractLMM_test:
    """Interaction test"""
    def __init__(self):
        pass

    def test_all(self):
        RV = False
        #print 'CInteractLMM IMPLEMENTED %s' % message(RV)


if __name__ == '__main__':
    unittest.main()
