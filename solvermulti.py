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

# calc for relevant archs
def relevantcustomers(a,N):
    ##define used customers for calculation
    inp = np.random.choice(np.arange(1, N.shape[0]), size=a, replace=False)
    inp = np.insert(inp, 0, 0)
    relN = N[0, :]