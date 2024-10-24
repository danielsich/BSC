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
## define empty array to fill in the distances
def customersize(a):
    nan_array = np.full((a+1,a+1),np.nan)
    return nan_array

print(customersize(5).shape)
##print(np.insert(np.random.choice(np.arange(1,100),size=10, replace = False)),0,0)


##set up customerarray
def dijin(a, Ns):
    ##define size of output array
    outp = customersize(a)
    
    ##define used customers for calculation
    inp = np.random.choice(np.arange(1, Ns.shape[0]), size=a, replace = False)
    inp = np.insert(inp, 0, 0)
    i = j = 0
    if i in inp.shape:
        if j in inp.shape:
            if(i != j):
                outp[i,j] = dist3deuclid(Ns[inp[i],:], Ns[inp[j], :])
            j += 1
                
        i += 1
    
    return outp





print(dijin(2,Ns))
print((dijin(2,Ns).shape[0]))