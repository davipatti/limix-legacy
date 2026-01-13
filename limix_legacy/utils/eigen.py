import numpy as np
import scipy.linalg as LA

def dS_dti(dM_dti,M=None,U=None,S=None):
    assert M is not None or U is not None, 'Specify either M or U and S'
    if U is None:
        U,S = LA.eigh(M) 
    RV = (U*np.dot(dM_dti,U)).sum(0)
    return RV
    
def dU_dti(dM_dti,M=None,U=None,S=None):
    assert M is not None or (U is not None and S is not None), 'Specify either M or U and S'
    if U is None or S is None:
        U,S = LA.eigh(M) 
    P  = dM_dti.shape[0]
    RV = np.zeros((P,P))
    for p in range(P):
        S1 = S[p]-S
        I  = abs(S1)>1e-4
        _U = U[:,I]
        _D = (S1[I]**(-1))[:,np.newaxis]
        pseudo = np.dot(_U,_D*_U.T)
        RV[:,p] = np.dot(pseudo,np.dot(dM_dti,U[:,p])) 
    return RV

