"""Datasets for LMM tests"""
import numpy as NP
import glob
import re
import os

filetype = ".txt"
#filetype = ".txt.gz"

def load(directory):
    stringpattern = "(.*)"+filetype
    pattern=re.compile(stringpattern)
    FL = glob.glob(os.path.join(directory,'*'+filetype))
    RV = {}
    for fn in FL:
        fn_ = os.path.split(fn)[-1] #only keep the filename
        name = pattern.match(fn_).group(1)
        RV[name] = NP.loadtxt(fn)
    return RV

def dump(R,directory):
    for r in list(R.keys()):
        fn = os.path.join(directory,r+filetype)
        NP.savetxt(fn,R[r])
    
    
