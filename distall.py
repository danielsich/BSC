# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:26:41 2024

@author: Daniel
"""

import numpy as np
## load in customer data
N = np.load('N.npy')

# define calculation for distance calculation
def dist3deuclid(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)

## define empty array to fill in the distances
def customersize(a):
    return np.zeros((a, a), dtype=float)

# define Matrix distance calculation for all
def distall(N):
    ##define size of output array
    outp = customersize(N.shape[0])
    i = 0
    j = 0
    for i in range(N.shape[0]):
        for j in range(N.shape[0]):
            if(i != j):
                c = N[i]
                d = N[j]
                outp[i, j] = dist3deuclid(c[:3], d[:3])
    np.save('Distall.npy', outp)
    return outp

abc = distall(N)
print(abc.shape)