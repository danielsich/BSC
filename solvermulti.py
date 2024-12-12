# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:24:04 2024

@author: Daniel
"""
import numpy as np
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

Distall = np.load('Distall.npy')
N = np.load('N.npy')

## set up the possible combinations
def tupls (a):
    arcs = [(i, j) for i in range(a) for j in range(a) if i != j]
    return np.array(arcs)

# string relevant customers
def relevantcusta(a):
    inp = np.random.choice(np.arange(1, N.shape[0]), size=a, replace=False)
    inp = np.insert(inp, 0, 0)
    return inp
# calc for relevant customers
def relevantcustomers(inp,N):
    relN = N[0, :]
    ##np.save('relN',inp)
    i = 0
    j = 0
    for i in range(len(inp)):
        j = inp[i]
        if (j != 0):
            relN = np.vstack((relN, N[j, :]))
    return relN
print(relevantcusta(5))
#define relvant Distnces
def relevantdistances(relevantcusta,Distall):
    return

