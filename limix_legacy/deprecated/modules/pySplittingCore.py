'''
Created on Jan 24, 2012

@author: johannes
'''
import sys
import numpy as NP
from . import lmm_fast
import limix_legacy.deprecated as dlimix_legacy

def is_leaf(node_ind, child_nodes):
    return child_nodes[node_ind] == 0

def get_ancestors(node_ind, node, parents):
    ancestors = NP.empty(NP.int_(NP.floor(NP.log2(node+1))), dtype='int')
    for i in NP.arange(ancestors.size):
        node_ind = parents[node_ind]
        ancestors[i] = node_ind
    return ancestors

def get_covariates(node_ind, node, parents, sample, start_index, end_index):
    node_indexes = get_ancestors(node_ind,node, parents)
    Covariates = NP.zeros((sample.size,node_indexes.size+1), order='F')
    #node itself

    Covariates[sample[start_index[node_ind]:end_index[node_ind]],0] = 1
    j = 1
    for node_ind in node_indexes:
        Covariates[sample[start_index[node_ind]:end_index[node_ind]],j] = 1
        j += 1
    return Covariates

def checkMaf(X, maf):
    Xmaf = (X==1).sum(axis=0)
    Iok = (Xmaf>=(maf*X.shape[0]))
    return NP.where(Iok)[0]

def scale_K(K, verbose=False):
    """scale covariance K such that it explains unit variance"""
    c = NP.sum((NP.eye(len(K)) - (1.0 / len(K)) * NP.ones(K.shape)) * NP.array(K))
    scalar = (len(K) - 1) / c
    if verbose:
        print(('Kinship scaled by: %0.4f' % scalar))
    K = scalar * K
    return K

def estimateKernel(X, maf):
    #1. maf filter
    Xpop = float(X[:,checkMaf(X, maf)]).copy()
    Xpop -= Xpop.mean(axis=0)
    Xpop /= Xpop.std(axis=0)
    Kpop  =  NP.dot(Xpop,Xpop.T)
    return scale_K(Kpop)

def estimate_bias(Uy, U, S, ldelta):
    UC = NP.dot(U.T,NP.ones_like(Uy))
    _, beta, _ = lmm_fast.nLLeval(ldelta, Uy[:,0], UC, S, MLparams=True)
    return beta[0]

def check_predictors(X, noderange, rmind):
    Xout = X[NP.ix_(noderange, rmind)]
    X_sum = NP.sum(Xout,0)
    indexes = (X_sum != Xout.shape[0]) & (X_sum != 0)
    return rmind[indexes]

def cpp_best_split_full_model(X, Uy, C, S, U, noderange, delta,
                              save_memory=False):
    """wrappe calling cpp splitting function"""

    m_best, s_best, left_mean, right_mean, ll_score =\
         dlimix_legacy.best_split_full_model(X, Uy, C, S, U, noderange, delta)
    return int(m_best), s_best, left_mean, right_mean, ll_score

def best_split_full_model(X,
                          Uy,
                          C,
                          S,
                          U,
                          noderange,
                          delta):
    mBest = -1
    sBest = -float('inf')
    score_best = -float('inf')
    left_mean = None
    right_mean = None
    ldelta = NP.log(delta)
    levels = list(map(NP.unique, X[noderange].T))
    feature_map = []
    s = []
    UXt = []
    cnt = 0
    for i in range(X.shape[1]):
        lev = levels[i]
        for j in range(lev.size-1):
            split_point = NP.median(lev[j:j+2])
            x = NP.int_(X[noderange,i] > split_point)
            UXt.append(NP.dot(U.T[:,noderange], x))
            feature_map.append(i)
            s.append(split_point)
            cnt += 1
    UXt = NP.array(UXt).T
    if UXt.size == 0: #predictors are homogeneous
        return mBest, sBest, left_mean, right_mean, score_best
    else:
        #print UXt
#         print X[noderange]
#         print ''
#         print ''
        # test all transformed predictors
        scores = -NP.ones(cnt)*float('inf')
        UC = NP.dot(U.T,C)
        ########################
        #finding the best split#
        ########################
        score_0 = lmm_fast.nLLeval(ldelta,Uy[:,0],UC,S)
        for snp_cnt in NP.arange(cnt):
            UX=NP.hstack((UXt[:,snp_cnt:snp_cnt+1], UC))
            scores[snp_cnt] = -lmm_fast.nLLeval(ldelta,Uy[:,0],UX,S)
            scores[snp_cnt] += score_0
        ############################
        ###evaluate the new means###
        ############################
        kBest = NP.argmax(scores)
        score_best = scores[kBest]
        sBest = s[kBest]
        if score_best > 0:
                sBest = s[kBest]
                score_best = scores[kBest]
                UX=NP.hstack((UXt[:,kBest:kBest+1], UC))
                _, beta,_ = lmm_fast.nLLeval(ldelta, Uy[:,0], UX, S, MLparams=True)
                mBest = feature_map[kBest]
                CX = NP.zeros_like(Uy)
                CX[noderange] = NP.int_(X[noderange,mBest:mBest+1] > sBest)
                C_new = NP.hstack((CX,C))
                mean = NP.dot(C_new,beta.reshape(beta.size, -1)) #TODO:is this the correct way?
                left_mean = ((mean[noderange])[CX[noderange]==0])[0]
                right_mean = ((mean[noderange])[CX[noderange]==1])[0]
        return mBest, sBest, left_mean, right_mean, score_best

if __name__== '__main__':
    node = 5
    # test building of covariates
    subsample = NP.arange(10,dtype='int')
    start_index =  [0,0,4,4,5]
    end_index = [10,4,10,5,10]
    node_labels = [0,1,2,5,6]
    parents = [0,0,0,2,2]
    node_ind = 4
    print((get_covariates(node_ind, node_labels[node_ind], parents, subsample, start_index, end_index)))
