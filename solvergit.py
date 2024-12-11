# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:36:11 2024

@author: Daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:41:56 2024

@author: Daniel
"""

import numpy as np
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from radial import betaa


load_dotenv()


## import relevant np.arrays
relN = np.load('relN.npy')
Dist = np.load('Dist.npy') 
Archs = np.load('Archs.npy') ## relevant Archs
lvl = np.load('lvl.npy') ## speed levels
tj0 = np.load('tj0.npy')
a_ij = np.load('aij.npy')

## customers
N0d = relN[1:] # information all customers

N = np.arange(relN.shape[0]) # All edges
N0 = N[1:] # alle customer edges
Nq = N.shape[0] # number edges
N0q = N0.shape[0] # number customers
qi = relN[:,3] #Demand Customer
ti = 1 # service time
ai = relN[:,4] # earliest time
bi = relN[:,5] # latest time

#print(qi)
##Vehicles
m = 9 #amount
K = np.arange(m)
Q = 3500  # Capacity
W = 3500  # curb weight 
p = 1
cfe = 2 #cost for fuel and emissions
BIGM = 999999999 ##bigM

options = {
    #configure()
    "WLSACCESSID" : os.getenv("WLSACCESSID"),
    "WLSSECRET" : os.getenv("WLSSECRET"), 
    "LICENSEID":2503389,
}
#print(options)
## start model

prp = gp.Model(env=gp.Env(params=options))
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

zfpd = quicksum(x[i, j] * Dist[i, j] for i in N for j in N)
zfpl = quicksum(a_ij[i, j] * Dist[i, j] * W *  x[i, j] for i in N for j in N) + quicksum(a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N)
zfpe = quicksum(a_ij[i, j] * Dist[i, j] * W *  x[i, j] for i in N for j in N) + quicksum(a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N) + quicksum(Dist[i, j] * betaa * (quicksum((lvl[r]**2) * z[i, j, r] for r in range(lvl.shape[0])))for i in N for j in N)
zfprp = quicksum(cfe * a_ij[i, j] * Dist[i, j] * W *  x[i, j] for i in N for j in N) + quicksum(cfe * a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N) + quicksum(cfe * Dist[i, j] * betaa * (quicksum((lvl[r]**2) * z[i, j, r] for r in range(lvl.shape[0])))for i in N for j in N) + quicksum(p * s[j] * x[j, 0] for j in N0)
prp.setObjective(zfprp, GRB.MINIMIZE)

## costraints
prp.addConstr(quicksum(x[0, j] for j in N0) == quicksum(x[j, 0] for j in N0), name="con10_better")

for i in N0:
    prp.addConstr(quicksum(x[i, j] for j in N) == 1, name=f"con11_{i}")
    
for j in N0:
    prp.addConstr(quicksum(x[i, j] for i in N) == 1, name=f"con12_{i}")

for i in N0:
    prp.addConstr(quicksum(f[j, i] for j in N) - quicksum(f[i, j] for j in N) == qi[i], name=f"con13_{i}")

for i, j in Archs:
    prp.addConstr(qi[j] * x[i,j] <= f[i, j], name=f"con14_low_{i}_{j}")
    prp.addConstr(f[i, j] <= (Q - qi[i])*x[i, j], name=f"con14_high_{i}_{j}")

for i in N:
    for j in N0:
        if i !=j:
            prp.addConstr(y[i] - y[j] + ti + quicksum((Dist[i, j] / lvl[r])* z[i, j, r] for r in range(lvl.shape[0])) <= BIGM * (1 - x[i, j]), name=f"con_15_{i}_{j}")

for i in N0:
    prp.addConstr(y[i] >= ai[i], name=f"con_16_low_{i}")
    
for i in N0:
    prp.addConstr(y[i] <= bi[i], name=f"con16_high_{i}")

for i in N0:
    prp.addConstr(y[i] + ti - s[i] + quicksum((Dist[i, 0] / lvl[r])* z[i, 0, r] for r in range(lvl.shape[0])) <= BIGM * (1 - x[i, 0]), name=f"con_17_{i}" )
  
for i, j in Archs:
    prp.addConstr(quicksum(z[i, j, r] for r in range(lvl.shape[0])) == x[i, j], name=f"con_18_{i}_{j}")

for j in N0:
    if j != 0:
        prp.addConstr((y[j] + ti + quicksum(tj0[j, r] * z[j, 0, r] for r in range(lvl.shape[0])))* x[j, 0] == s[j], name=f"constr_22{j}")

for i,j in Archs:
    prp.addConstr(f[i,j] >= 0, name=f"con_20_{i}_{j}")
    
for i in N0:
    for j in N0:
        if i!= j:
            prp.addConstr(x[i, j] + x[j, i] <= 1, name='subtourbreaking')
##set params
prp.setParam('TimeLimit', 10000)
prp.setParam('OutputFlag', 0)  

prp.update()
prp.optimize()
xVar = prp.getAttr('x', x)
sout = prp.getAttr('x', s)
yi = prp.getAttr('x', y)
speed = prp.getAttr('x', z)
flow = prp.getAttr('x', f)
relflow = tuplelist((i, j) for i,j in flow.keys() if flow[i,j] > 0)
#zout = prp.getAttr('x', z)
xrel = tuplelist((i, j) for i,j in xVar.keys() if xVar[i,j] == 1)
np.save('xrel.npy', xrel)
print(relflow)
print('--------')
#print(Archs)
print(xrel)

def discoor(abc, xrel, yi, speed, lvl):
    x = abc[:, 0]
    y = abc[:, 1]
    coordinates = abc[:, :2]
    
    plt.figure(figsize=(16,12))
    
    # Define a list of colors to use for different routes
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_index = 0
    
    # Identify routes starting from the depot (index 0)
    tours = []
    for start, end in xrel:
        if start == 0:
            tour = [(start, end)]
            next_node = end
            while next_node != 0:
                for (s, e) in xrel:
                    if s == next_node:
                        tour.append((s, e))
                        next_node = e
                        break
            tours.append(tour)
    
    # Plot each tour with a unique color and display the chosen speed
    for tour in tours:
        color = colors[color_index % len(colors)]
        color_index += 1
        for (start, end) in tour:
            start_coord = coordinates[start]
            end_coord = coordinates[end]
            plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], f'{color}o-')
            
            # Find the chosen speed for the arch
            chosen_speed = None
            for r in range(len(lvl)):
                if speed[start, end, r] > 0.5:
                    chosen_speed = lvl[r]
                    break
            if chosen_speed is not None:
                mid_x = (start_coord[0] + end_coord[0]) / 2
                mid_y = (start_coord[1] + end_coord[1]) / 2
                plt.text(mid_x, mid_y, f'{chosen_speed:.2f}', fontsize=10, ha='center', color='blue')
    
    for idx, (x, y) in enumerate(coordinates):
        plt.text(x, y, f'{idx} ({yi[idx]:.2f})', fontsize=12, ha='right', color='red')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Routes, Visit Times, and Chosen Speeds")
    plt.grid()
    plt.show()

# Example usage
discoor(relN, xrel, yi, speed, lvl)

def getTour(xrel):
    for x1 in xrel:
        xl = x1[0]
        xr = x1[1]
        print(f"Leftval: {xl}, rightval: {xr}")

#getTour(xrel)
    
def check_sout_greater_than_yi(sout, yi, xrel):
    """
  Check if s is bigger then y to check if model is true
    """
    # Find the last customers before returning to the depot (node 0)
    last_customers = [i for (i, j) in xrel if j == 0]
    
    # Check if sout is greater than yi for each last customer
    for customer in last_customers:
        if sout[customer] <= yi[customer]:
            return False
    return True


res = check_sout_greater_than_yi(sout, yi, xrel)
print("s is greater than y for all last customers before returning to the depot:", res)
    