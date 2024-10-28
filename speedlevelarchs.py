# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 05:54:07 2024

@author: Daniel
"""
## import numpy
import numpy as np

## set seed
np.random.seed(37)

## load archs
Archs = np.load('Archs.npy') ## relevant Archs

##define Vr
def levels(low, high, level):
    diff = high - low
    ranges = diff/level
    lvl = [(low + i * ranges + 0.5 * ranges) for i in range(level)]
    return np.array(lvl)
'''
def speedarchs(low, high, level, Arch):
    outp = for i in range(len(Arch)):
    '''    
        
