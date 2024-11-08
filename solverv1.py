# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:41:56 2024

@author: Daniel
"""

import numpy as np
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt




## import relevant np.arrays
relN = np.load('relN.npy')
Dist = np.load('Dist.npy') 
Archs = np.load('Archs.npy') ## relevant Archs
lvl = np.load('lvl.npy') ## speed levels
tj0 = np.load('tj0.npy')

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

print(N0q)
##Vehicles
m = 10 #amount
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
    
s = {}
for i in range(Nq):
   s[i] = prp.addVar(vtype= GRB.CONTINUOUS)

zf = quicksum(x[i, j] * Dist[i, j] for i in N for j in N)
prp.setObjective(zf, GRB.MINIMIZE)

## costraints
prp.addConstr(quicksum(x[0, j] for j in N0) == quicksum(x[j, 0] for j in N0), name="con10_better")

for i in N0:
    prp.addConstr(quicksum(x[i, j] for j in N) == 1, name=f"con11_{i}")
    prp.addConstr(quicksum(x[j, i] for j in N) == 1, name=f"con12_{i}")

for i in N0:
    prp.addConstr(quicksum(f[j, i] for j in N)-quicksum(f[i, j] for j in N) == qi[i], name=f"con13_{i}")

for i, j in Archs:
    prp.addConstr(qi[j] * x[i,j] <= f[i, j], name=f"con14_low_{i}_{j}")
    prp.addConstr(f[i, j] <= (Q - qi[i]*x[i, j]), name=f"con14_high_{i}_{j}")

for i in N:
    for j in N0:
        if i !=j:
            prp.addConstr(y[i] - y[j] + ti + quicksum((Dist[i, j] / lvl[r])* z[i, j, r] for r in range(lvl.shape[0])) <= BIGM * (1 - x[i, j]), name=f"con_15_{i}_{j}")

for i in N0:
    prp.addConstr(y[i] >= ai[i], name=f"con_16_low_{i}")
    prp.addConstr(y[i] <= bi[i], name=f"con16_high_{i}")
    
for i in N0:
    prp.addConstr(y[i] + ti - s[i-1] + quicksum((Dist[i, 0] / lvl[r])* z[i, 0, r] for r in range(lvl.shape[0])) <= BIGM * (1 - x[i, 0]), name=f"con_17_{i}" )
  
for i, j in Archs:
    prp.addConstr(quicksum(z[i, j, r] for r in range(lvl.shape[0])) == x[i, j], name=f"con_18_{i}_{j}")

for j in N:
    if j != 0:
        prp.addConstr((y[j] + ti + quicksum(tj0[j, r] * z[j, 0, r] for r in range(lvl.shape[0])))* x[j, 0] == s[j], name=f"constr_22{j}")

for i in N0:
    prp.addConstr(quicksum(x[i, j] + x[j, i] for j in N0) <= 1, name='subtourbreaking')
##set params
prp.setParam('TimeLimit', 100)
prp.setParam('OutputFlag', 0)  

prp.update()
prp.optimize()
xVar = prp.getAttr('x', x)
sout = prp.getAttr('x', s)
yi = prp.getAttr('x', y)
#zout = prp.getAttr('x', z)
xrel = tuplelist((i, j) for i,j in xVar.keys() if xVar[i,j] == 1)
np.save('xrel.npy', xrel)
print(xrel)


def discoor(abc):
    x = abc[:, 0]
    y = abc[:, 1]
    
    plt.scatter(x, y)
    plt.show()
    
discoor(relN)

def getTour(xrel):
    for x1 in xrel:
        xl = x1[0]
        xr = x1[1]
        print(f"Leftval: {xl}, rightval: {xr}")

#getTour(xrel)
    
'''
print(xVar)
print("---------------------------")
print(sout)
print("---------------------------")
print(yi)
print("---------------------------")
#print(zout)
print("---------------------------")
print("Zielfunktionswert:",prp.objVal)

route = []


for i in xVar:
    for j in xVar:
        
        if(xVar.keys(i,j) > 0.5):
            print(f"{i}->{j}")
            i = j
            j = 0
        '''
    
    