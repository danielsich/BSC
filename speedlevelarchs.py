# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 05:54:07 2024

@author: Daniel
"""
## import numpy
import numpy as np

## set seed
np.random.seed(37)

## load distances
##Archs = np.load('Archs.npy') ## relevant Archs
Dist = np.load('Dist.npy') 

##define averagespeed for the relevant speed levels
def levels(low, high, level):
    diff = high - low
    ranges = diff/level
    lvl = [(low + i * ranges + 0.5 * ranges) for i in range(level)]
    np.save('lvl.npy',lvl)
    return np.array(lvl)

a = levels(40, 100, 60)
print(a.ndim)
print(Dist)
print("--------------------------------")
##print(Dist[0,:])

def tj0(levels, Dist):
    assert Dist.ndim == 2
    assert levels.ndim == 1
    trvlt = np.zeros((Dist.shape[0], levels.shape[0]))
    for i in range(Dist.shape[0]):
        for j in range(len(levels)):
            trvlt[i, j] = Dist[i, 0] / levels[j]
            
    np.save('tj0.npy',trvlt)
    return trvlt
print(tj0(a, Dist))
            
    
    
    

'''
def speedarchs(low, high, level, Arch):
    outp = for i in range(len(Arch)):
    '''    
        
