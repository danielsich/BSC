# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:23:08 2024

@author: Daniel
"""

import numpy as np

## set seed
np.random.seed(37)

## load in relevant customer data
relN = np.load('relN.npy')
##print(relN)

def ang(N):
    n = N.shape[0]
    ang = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if(i != j):
                dot = np.dot(N[i], N[j])
                magi = np.linalg.norm(N[i]) 
                magj = np.linalg.norm(N[j])
                cos_theta = dot / (magi * magj)
                cos_theta = np.clip(cos_theta,-1.0, 1.0)
                ang[i, j] = np.arccos(cos_theta)
    
    np.save('ang.npy',ang)
    return ang
angl = ang(relN[:, :3])
print(angl)