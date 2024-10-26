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
N0 = relN[1:]
print(N0)
K = [0,1]
