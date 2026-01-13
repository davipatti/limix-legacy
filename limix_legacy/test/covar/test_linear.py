"""LMM testing code"""
import unittest
import numpy as NP
import pdb
import limix_legacy.deprecated as dlimix_legacy
from .covar import Acovar_test

class CCovLinearISO_test(unittest.TestCase,Acovar_test):
    """test class for CCovLinearISO"""
    def setUp(self):
        NP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=NP.random.rand(self.n,self.n_dim)
        self.C = dlimix_legacy.CCovLinearISO(self.n_dim)
        self.name = 'CCovLinearISO'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=NP.exp(NP.random.randn(self.n_params))
        self.C.setParams(params)

class CCovLinearARD_test(unittest.TestCase,Acovar_test):
    """test class for CCovLinearARD"""
    def setUp(self):
        NP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=NP.random.rand(self.n,self.n_dim)
        self.C = dlimix_legacy.CCovLinearARD(self.n_dim)
        self.name = 'CCovLinearARD'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=NP.exp(NP.random.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
