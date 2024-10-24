# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:49:46 2024

@author: Daniel
"""

import numpy as np

## set seed
np.random.seed(37)

## load in customer data
N = np.load('N.npy')

## only keep the first three collums
Ns = N[:,:3] 

## change hight to Kilometers
Ns[:, 2] = Ns[:, 2] / 1000
np.set_printoptions(suppress=True, precision = 8)
##print(Ns)

# define calculation for distance calculation
def dist3deuclid(a, b):
    a = np.array(a)
    b = np.array(b)
    
    distance = np.linalg.norm(a-b)
    
    return distance
a = Ns[0, :]
b = Ns[1, :] 
print(dist3deuclid(a, b))

def customersize(a):
    nan_array = np.full((a+1,a+1),np.nan)
    return nan_array