"""CLkelihoodISO testing code"""
import unittest
import numpy as NP
import pdb
import limix_legacy.deprecated as dlimix_legacy
from .covar import Acovar_test

class CLikelihoodISO_test(unittest.TestCase,Acovar_test):
    """test class for CLikelihoodISO"""
    def setUp(self):
        NP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=NP.random.rand(self.n,self.n_dim)
        self.C = dlimix_legacy.CLikNormalIso()
        self.name = 'CLikNormalIso'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        params=NP.exp(NP.random.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
