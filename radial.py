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
    ang = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(i != j):
                dot = np.dot(N[i], N[j])
                magi = np.linalg.norm(N[i]) 
                magj = np.linalg.norm(N[j])
                cos_theta = dot / (magi * magj)
                #cos_theta = np.clip(cos_theta,-1.0, 1.0)
                ang[i, j] = np.arccos(cos_theta)
    
    np.save('ang.npy',ang)
    return ang
angl = ang(relN[:, :3])

def aij(a, Cr, angl):
    g = 9.81 #gravity
    a_ij = np.zeros_like(angl)
    
    for i in range(angl.shape[0]):
        for j in range(angl.shape[1]):
            a_ij[i, j] = a + g * np.sin(angl[i,j]) + g * Cr * np.cos(angl[i, j])
            
    np.save('aij.npy', a_ij)
    return a_ij


a_ij = aij(0, 0.01, angl)

def b(cd,A,p):
    return 0.5 * cd * A * p  

b = b(0.7, 5, 1.2041)
print(b)