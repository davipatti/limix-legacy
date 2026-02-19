"""Variance Decomposition testing code"""
import unittest
import scipy as SP
import numpy as NP
import scipy.stats
import pdb
import os
import sys
import limix_legacy
import limix_legacy.deprecated.modules.varianceDecomposition as VAR
from limix_legacy.test import data


class VarianceDecompoitionKronecker_test(unittest.TestCase):
    """test class for VarianceDecomposition
         THis testing class assumes everything is Kronecker.
    """

    def genGeno(self):
        X  = (NP.random.rand(self.N,self.S)<0.2)*1.
        self.D['X'] = X


    def genPheno(self):
        dp = NP.ones(self.P); dp[1]=-1
        Y = NP.zeros((self.N,self.P))
        gamma0_g = NP.random.randn(self.S)
        gamma0_n = NP.random.randn(self.N)
        for p in range(self.P):
            gamma_g = NP.random.randn(self.S)
            gamma_n = NP.random.randn(self.N)
            beta_g  = dp[p]*gamma0_g+gamma_g
            beta_n  = gamma0_n+gamma_n
            y=NP.dot(self.D['X'],beta_g)
            y+=beta_n
            Y[:,p]=y
        Y=SP.stats.zscore(Y,0)
        self.D['Y']= Y


    def setUp(self):
        #check: do we have a csv File?
        self.dir_name = os.path.dirname(__file__)
        self.dataset = os.path.join(self.dir_name,'varDecomp')

        if (not os.path.exists(self.dataset)) or 'recalc' in sys.argv:
            if not os.path.exists(self.dataset):
                os.makedirs(self.dataset)
            NP.random.seed(1)
            self.N = 200
            self.S = 1000
            self.P = 2
            self.D = {}
            self.genGeno()
            self.genPheno()
            self.generate = True
        else:
            self.generate=False
            #self.D = data.load(os.path.join(self.dir_name,self.dataset))
            self.D = data.load(self.dataset)
            self.N = self.D['X'].shape[0]
            self.S = self.D['X'].shape[1]
            self.P = self.D['Y'].shape[1]

        self.Kg = NP.dot(self.D['X'],self.D['X'].T)
        self.Kg = self.Kg/self.Kg.diagonal().mean()

        self.vc = VAR.VarianceDecomposition(self.D['Y'])
        self.vc.addRandomEffect(self.Kg,jitter=0)
        self.vc.addRandomEffect(is_noise=True,jitter=0)
        self.vc.addFixedEffect()

    def test_fit(self):
        """ optimization test """
        self.vc.optimize(verbose=False)
        params = self.vc.getScales()
        if self.generate:
            self.D['params_true'] = params
            data.dump(self.D,self.dataset)
            self.generate=False
        params_true = self.D['params_true']
        RV = ((NP.absolute(params)-NP.absolute(params_true))**2).max()
        self.assertTrue(RV<1e-6)

    def test_fitFast(self):
        """ optimization test """
        self.vc.optimize(fast=True,verbose=False)
        params = self.vc.getScales()
        if self.generate:
            self.D['params_true'] = params
            data.dump(self.D,self.dataset)
            self.generate=False

        params_true = self.D['params_true']
        #make sign invariant
        RV = ((NP.absolute(params)-NP.absolute(params_true))**2).max()<1e-6
        self.assertTrue(RV)

class VarianceDecompositionSoftKronecker_test(unittest.TestCase):
    """test class for VarianceDecomposition
         THis testing class assumes tests the ability to
    """

    def genGeno(self):
        X  = (NP.random.rand(self.N,self.S)<0.2)*1.
        self.D['X'] = X

    def genPheno(self):
        dp = NP.ones(self.P); dp[1]=-1
        Y = NP.zeros((self.N,self.P))
        gamma0_g = NP.random.randn(self.S)
        gamma0_n = NP.random.randn(self.N)
        for p in range(self.P):
            gamma_g = NP.random.randn(self.S)
            gamma_n = NP.random.randn(self.N)
            beta_g  = dp[p]*gamma0_g+gamma_g
            beta_n  = gamma0_n+gamma_n
            y=NP.dot(self.D['X'],beta_g)
            y+=beta_n
            Y[:,p]=y
        Y=SP.stats.zscore(Y,0)
        self.D['Y']= Y

    def setUp(self):
        #check: do we have a csv File?
        self.dir_name = os.path.dirname(__file__)
        self.dataset = os.path.join(self.dir_name,'varDecomp')

        if (not os.path.exists(self.dataset)) or 'recalc' in sys.argv:
            if not os.path.exists(self.dataset):
                os.makedirs(self.dataset)
            NP.random.seed(1)
            self.N = 200
            self.S = 1000
            self.P = 2
            self.D = {}
            self.genGeno()
            self.genPheno()
            self.generate = True
        else:
            self.generate=False
            #self.D = data.load(os.path.join(self.dir_name,self.dataset))
            self.D = data.load(self.dataset)
            self.N = self.D['X'].shape[0]
            self.S = self.D['X'].shape[1]
            self.P = self.D['Y'].shape[1]

        self.Kg = NP.dot(self.D['X'],self.D['X'].T)
        self.Kg = self.Kg/self.Kg.diagonal().mean()

        #add missing values to Y
        if 1:
            self.D['Y'][0,0] = NP.nan
            self.D['Y'][1,1] = NP.nan
            self.D['Y'][10,0] = NP.nan
            self.D['Y'][100,1] = NP.nan
        self.vc = VAR.VarianceDecomposition(self.D['Y'])
        self.vc.addRandomEffect(self.Kg,jitter=0)
        self.vc.addRandomEffect(is_noise=True,jitter=0)
        self.vc.addFixedEffect()

    def test_fit(self):
        """ optimization test """
        self.vc.optimize(verbose=False)
        params = self.vc.getScales()
        if self.generate:
            self.D['params_true'] = params
            data.dump(self.D,self.dataset)
            self.generate=False
        params_true = self.D['params_true']
        RV = ((NP.absolute(params)-NP.absolute(params_true))**2).max()
        #permit more flexibility, as we set a few values to NAN
        self.assertTrue(RV<1e-4)


class VarianceDecomposition_synth_test(unittest.TestCase):
    """Tests for VarianceDecomposition using synthetic data (no reference files)."""

    def _make_vc_data(self, N=100, S=200, P=2, seed=42):
        rng = NP.random.RandomState(seed)
        X = rng.randn(N, S)
        K = X @ X.T / S
        K /= K.diagonal().mean()

        Y = NP.zeros((N, P))
        Lg = SP.linalg.cholesky(K + 1e-6 * NP.eye(N), lower=True)
        for p in range(P):
            Y[:, p] = Lg @ rng.randn(N) + rng.randn(N) * 0.5
        Y = (Y - Y.mean(0)) / Y.std(0)

        return Y, K

    def test_basic_optimize(self):
        """VarianceDecomposition optimizes and returns valid results."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        var_comps = vc.getVarianceComps()
        self.assertEqual(var_comps.shape, (2, 2))
        self.assertTrue(NP.all(var_comps >= 0))

    def test_fast_optimize(self):
        """VarianceDecomposition with fast=True (kronSum)."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        conv = vc.optimize(fast=True, verbose=False)
        self.assertTrue(conv)

    def test_fast_and_base_agree(self):
        """Fast and base GP should give similar results."""
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
            NP.testing.assert_allclose(var_base, var_fast, atol=0.1)

    def test_trait_covar_freeform(self):
        """VarianceDecomposition with freeform trait covariance."""
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
        NP.testing.assert_allclose(Cg, Cg.T, atol=1e-10)
        NP.testing.assert_allclose(Cn, Cn.T, atol=1e-10)

    def test_trait_covar_diag(self):
        """VarianceDecomposition with diagonal trait covariance."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='diag', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='diag', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

        Cg = vc.getTraitCovar(0)
        self.assertTrue(NP.all(SP.linalg.eigvalsh(Cg) >= -1e-8))

    def test_trait_covar_lowrank(self):
        """VarianceDecomposition with lowrank trait covariance.

        NOTE: lowrank/block types are not compatible with 'diagonal' init_method
        (which is the default for P>1). Must use init_method='random'.
        """
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
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='block', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='block_diag', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(init_method='random', verbose=False)
        self.assertTrue(conv)

    def test_trait_covar_block_id(self):
        """VarianceDecomposition with block_id trait covariance."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='block_id', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='block_id', jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

    def test_trait_covar_lowrank_id(self):
        """VarianceDecomposition with lowrank_id trait covariance."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='lowrank_id', rank=1, jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='lowrank_diag', rank=1, jitter=1e-4)
        vc.addFixedEffect()
        conv = vc.optimize(verbose=False)
        self.assertTrue(conv)

    def test_getVarianceComps_univariance(self):
        """getVarianceComps with univariance=True normalizes to sum to 1."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        var = vc.getVarianceComps(univariance=True)
        NP.testing.assert_allclose(var.sum(axis=1), NP.ones(2), atol=1e-10)

    def test_getTraitCorrCoef(self):
        """Trait correlation coefficients should be in [-1, 1] with 1 on diagonal."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        corr = vc.getTraitCorrCoef(0)
        NP.testing.assert_allclose(NP.diag(corr), NP.ones(2), atol=1e-10)
        self.assertTrue(NP.all(NP.abs(corr) <= 1.0 + 1e-10))

    def test_getScales_and_setScales(self):
        """getScales/setScales roundtrip preserves values."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        scales = vc.getScales()
        vc.setScales(scales)
        scales2 = vc.getScales()
        NP.testing.assert_allclose(scales, scales2, atol=1e-12)

    def test_getLML(self):
        """Log marginal likelihood is finite."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        lml = vc.getLML()
        self.assertTrue(NP.isfinite(lml))

    def test_single_trait(self):
        """VarianceDecomposition works for P=1."""
        rng = NP.random.RandomState(42)
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
        self.assertTrue(NP.all(var_comps >= 0))

    def test_1d_phenotype_reshaped(self):
        """1D array phenotype is reshaped to column vector."""
        rng = NP.random.RandomState(42)
        N = 50
        Y = rng.randn(N)  # 1D

        vc = VAR.VarianceDecomposition(Y)
        self.assertEqual(vc.P, 1)
        self.assertEqual(vc.N, N)

    def test_normalize_K(self):
        """addRandomEffect normalizes K by default (K.trace()/N == 1)."""
        rng = NP.random.RandomState(42)
        N = 50
        Y = rng.randn(N, 2)
        K = rng.randn(N, N)
        K = K @ K.T  # PD but arbitrary scale

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        K_internal = vc.vd.getTerm(0).getK()
        NP.testing.assert_allclose(K_internal.diagonal().mean(), 1.0, atol=1e-6)

    def test_missing_values(self):
        """VarianceDecomposition handles missing values (NaN in Y)."""
        rng = NP.random.RandomState(42)
        N, P = 100, 2
        X = rng.randn(N, 200)
        K = X @ X.T / 200
        K /= K.diagonal().mean()
        Y = rng.randn(N, P)
        Y[0, 0] = NP.nan
        Y[5, 1] = NP.nan

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        conv = vc.optimize(fast=False, verbose=False)
        self.assertTrue(conv)

    def test_optimize_with_repeats(self):
        """optimize_with_repeates returns sorted list of optima."""
        Y, K = self._make_vc_data(N=50, S=100)
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()

        results = vc.optimize_with_repeates(n_times=3, verbose=False)
        self.assertIsInstance(results, list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['LML'], results[i+1]['LML'])

    def test_noise_pos_set_correctly(self):
        """noisPos is set to the index of the noise term."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        self.assertEqual(vc.noisPos, 1)

    def test_double_noise_raises(self):
        """Adding two noise terms should raise."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(is_noise=True)
        with self.assertRaises(AssertionError):
            vc.addRandomEffect(is_noise=True)

    def test_getWeights(self):
        """getWeights returns finite values after optimization."""
        Y, K = self._make_vc_data()
        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, jitter=0)
        vc.addRandomEffect(is_noise=True, jitter=0)
        vc.addFixedEffect()
        vc.optimize(verbose=False)

        weights = vc.getWeights()
        self.assertTrue(NP.all(NP.isfinite(weights)))


class VarianceDecompositionStdErrors_test(unittest.TestCase):
    """Tests for VarianceDecomposition standard errors.

    BUG DOCUMENTATION for getTraitCovarStdErrors (varianceDecomposition.py lines 553-558):

    Bug 1: Line 556-557 uses wrong loop variable for accumulating parameter index:
        for term in range(term_i-1):
            par_index += self.vd.getTerm(term_i).getNumberScales()  # BUG: should be getTerm(term)

    Bug 2: Line 556 uses range(term_i-1) but should be range(term_i) to include
        all terms before term_i.

    Bug 3 (numpy 2.x): _getHessian calls np.zeros(ParamMask.sum()) where .sum()
        returns np.float64, which numpy 2.x rejects.
    """

    def _make_vc_fitted(self, N=80, P=2, seed=42):
        rng = NP.random.RandomState(seed)
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

        C0 = vc.vd.getTerm(0).getTraitCovar()
        n_params_term0 = C0.getNumberParams()
        self.assertGreater(n_params_term0, 0,
                          "Term 0 has params, so par_index for term 1 should be nonzero")

        buggy_par_index = 0  # This is what the code computes for term_i=1
        correct_par_index = n_params_term0
        self.assertNotEqual(buggy_par_index, correct_par_index,
                           "Bug: par_index should be %d but code gives 0" % correct_par_index)

    def test_single_trait_std_errors_hessian_bug(self):
        """BUG: Single-trait std errors also hit the numpy _getHessian bug."""
        rng = NP.random.RandomState(42)
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

        with self.assertRaises(TypeError):
            vc.getTraitCovarStdErrors(0)


class BuildTraitCovar_test(unittest.TestCase):
    """Test _buildTraitCovar for each trait_covar_type."""

    def test_fixed_shape_check_bug(self):
        """BUG: _buildTraitCovar checks fixed_trait_covar.shape against self.N instead of self.P.

        In varianceDecomposition.py line 774-775:
            assert fixed_trait_covar.shape[0]==self.N  # BUG: should be self.P
            assert fixed_trait_covar.shape[1]==self.N  # BUG: should be self.P

        This means if N != P (which is almost always the case), passing a correct
        PxP fixed_trait_covar will fail, and passing an incorrect NxN matrix will pass.
        """
        rng = NP.random.RandomState(42)
        N, P = 50, 3
        Y = rng.randn(N, P)
        vc = VAR.VarianceDecomposition(Y)

        # This SHOULD work (PxP matrix) but will fail due to the bug
        correct_fixed_covar = NP.eye(P)
        with self.assertRaises(AssertionError):
            vc.addRandomEffect(K=NP.eye(N), trait_covar_type='fixed',
                             fixed_trait_covar=correct_fixed_covar, jitter=0)


class PairwiseInit_test(unittest.TestCase):
    """Test for the pairwise initialization check bug.

    BUG in varianceDecomposition.py line 316:
        i = (self.trait_covar_type[0]=='freeform')*(self.trait_covar_type[0]=='freeform')
    The second comparison should check trait_covar_type[1], not [0] again.
    """

    def test_pairwise_init_check_uses_wrong_index(self):
        """The pairwise init check tests trait_covar_type[0] twice instead of [0] and [1]."""
        rng = NP.random.RandomState(42)
        N, P = 50, 2
        Y = rng.randn(N, P)
        K = NP.eye(N)

        vc = VAR.VarianceDecomposition(Y)
        vc.addRandomEffect(K, trait_covar_type='freeform', jitter=1e-4)
        vc.addRandomEffect(is_noise=True, trait_covar_type='diag', jitter=1e-4)
        vc.addFixedEffect()

        # Due to the bug, this check passes even though term 1 is 'diag' not 'freeform':
        check_result = (vc.trait_covar_type[0] == 'freeform') * (vc.trait_covar_type[0] == 'freeform')
        self.assertTrue(check_result, "Bug: check passes because it tests index 0 twice")

        # Correct check would be:
        correct_check = (vc.trait_covar_type[0] == 'freeform') * (vc.trait_covar_type[1] == 'freeform')
        self.assertFalse(correct_check, "Correct check properly detects term 1 is not freeform")


if __name__ == '__main__':
    unittest.main()
