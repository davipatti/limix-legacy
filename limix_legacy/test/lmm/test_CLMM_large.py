"""LMM testing code"""
import unittest
import numpy as NP
import pdb
import limix_legacy
import limix_legacy.deprecated as dlimix_legacy
from limix_legacy.test import data
import os


class CLMM_test_large(unittest.TestCase):
    """test class for CLMM"""

    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(os.path.realpath(__file__))

    def test_lmm1(self):
        """basic test, comapring pv"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #make vllarg X. This needs to be changed later
            NL = 1000
            self.NL = NL
            X = NP.tile(D['X'],(1,self.NL))
            lmm = dlimix_legacy.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(X)
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv = lmm.getPv().ravel()
            BetaSte = lmm.getBetaSNPste().ravel()
            Beta = lmm.getBetaSNP()
            D2pv= (NP.log10(pv)-NP.log10(NP.tile(D['pv'],self.NL))**2)
            # D2Beta= (Beta-NP.tile(D['Beta'],self.NL))**2
            # D2BetaSte = (BetaSte-NP.tile(D['BetaSte'],self.NL))**2
            RV = NP.sqrt(D2pv.mean())<1E-6
            # RV = RV & (D2Beta.mean()<1E-6)
            # RV = RV & (D2BetaSte.mean()<1E-6)
            self.assertTrue(RV)



if __name__ == '__main__':
    unittest.main()
