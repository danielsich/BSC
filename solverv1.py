# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:41:56 2024

@author: Daniel
"""

import numpy as np
import gurobipy as gp
from gurobipy import *


## set seed
np.random.seed(37)

## import relevant np.arrays
relN = np.load('relN.npy')
Dist = np.load('Dist.npy') 
##print(relN)
## customers
N0d = relN[1:]
##print(N0d)


N = np.arange(relN.shape[0]) ## Alle Knoten
N0 = N[1:] ## alle Kundenknoten
Nq = N.shape[0] ## anzahl Knoten
N0q = N0.shape[0] ## anzahl Kunden
qi = relN[:,3]
ti = 10
ai = relN[:,4]
bi = relN[:,5]
##fahrzeuge
m = 2 #anzahl
K = np.arange(m)
Q = 3500
print(relN)
print(N0q)



