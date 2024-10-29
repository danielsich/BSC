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
Archs = np.load('Archs.npy') ## relevant Archs
lvl = np.load('lvl.npy') ## speed levels
##print(Archs)
## customers
N0d = relN[1:] # information all customers

N = np.arange(relN.shape[0]) # All edges
N0 = N[1:] # alle customer edges
Nq = N.shape[0] # number edges
N0q = N0.shape[0] # number customers
qi = relN[:,3] #Demand Customer
ti = 10 # service time
ai = relN[:,4] # earliest time
bi = relN[:,5] # latest time

##Vehicles
m = 3 #amount
K = np.arange(m)
Q = 3500  # Capacity

BIGM = 999999 ##bigM

## start model
prp = Model()
x = {} ## binary decison variable 1 if the arch is driven
for i in range(Nq):
    for j in range(Nq):
        x[i, j] = prp.addVar(vtype= GRB.BINARY)
        
z = {} ## available speedlevels for each arch
for i in range(Nq):
    for j in range (Nq):
        for r in range(lvl.shape[0]):
            z[i, j, r] = prp.addVar(vtype = GRB.BINARY)

f = {} ## amout of product flowing through each arch
for i in range(Nq):
    for j in range(Nq):
        f[i, j] = prp.addVar(vtype= GRB.CONTINUOUS)

y = {} ## time at wich node i is visited
for i in range(Nq):
    y[i] = prp.addVar(vtype= GRB.CONTINUOUS)

zf = quicksum(x[i, j] * Dist[i, j] for i in N for j in N)
prp.setObjective(zf, GRB.MINIMIZE)

## costraints
prp.addConstr(quicksum(x[0, j] for j in N0) == m, name="con10")

for i in N0:
    prp.addConstr(quicksum(x[i, j] for j in N0) == 1, name=f"con11_{i}")
    prp.addConstr(quicksum(x[j, i] for j in N0) == 1, name=f"con12_{i}")

for i in N0:
    prp.addConstr(quicksum(f[j, i] for j in N)-quicksum(f[i, j] for j in N) == qi[i], name=f"con13_{i}")

for i, j in Archs:
    prp.addConstr(qi[j] * x[i,j] <= f[i, j], name=f"con14_low_{i}_{j}")
    prp.addConstr(f[i, j] <= (Q - qi[i]*x[i, j]), name=f"con14_high_{i}_{j}")


for i, j in Archs:
    prp.addConstr(quicksum(z[i, j, r] for r in range(lvl.shape[0])) == x[i, j], name=f"con_18_{i}_{j}")
##set params
prp.setParam('TimeLimit', 5)
prp.update()
prp.optimize()
