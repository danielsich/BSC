# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:49:46 2024

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt

## set seed
np.random.seed(37)

## load in customer data
N = np.load('N.npy')

## only keep the first three collums
##Ns = N[:,:3] 

## change hight to Kilometers
###Ns[:, 2] = Ns[:, 2] / 1000
np.set_printoptions(suppress=True, precision = 8)
##print(Ns)

# define calculation for distance calculation
def dist3deuclid(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a-b)
aa = 0 
bb = 1
a = N[aa, :3]
b = N[bb, :3] 
##print(dist3deuclid(a, b))
## define empty array to fill in the distances
def customersize(a):
    return np.zeros((a+1, a+1), dtype=float)

print(customersize(5).shape)
##print(np.insert(np.random.choice(np.arange(1,100),size=10, replace = False)),0,0)

## get the possible combinations
def tupls (a):
    arcs = [(i, j) for i in range(a) for j in range(a) if i != j]
    return np.array(arcs)
    
##set up customerarray
def dijin(a,Ns):
    ##define size of output array
    outp = customersize(a)
    
    ##define used customers for calculation
    inp = np.random.choice(np.arange(1, Ns.shape[0]), size=a, replace = False)
    inp = np.insert(inp, 0, 0)
    relN = Ns[0,:]
    
    ##np.save('relN',inp)
    i = 0
    j = 0
    c = 0 
    d = 0
    for i in range(len(inp)):
        j = inp[i]
        if(j != 0):
            relN = np.vstack((relN,Ns[j, :]))
        
    
    i = 0
    j = 0
    for i in range(len(inp)):
        for j in range(len(inp)):
            if(i != j):
                c = inp[i]
                d = inp[j]
                outp[i,j] = dist3deuclid(Ns[c,:3], Ns[d, :3])
    Archs = tupls(a + 1)
    np.save('Archs.npy', Archs)
    np.save('relN.npy', relN)  
    np.save('Dist.npy', outp)        
    return outp



    

abc = dijin(25,N)
##np.save('Dist.npy',abc)
print(abc)


