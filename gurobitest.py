# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("example")

# Create decision variables
x = model.addVar(vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(vtype=GRB.CONTINUOUS, name="y")

# Set objective: Maximize 3x + 4y
model.setObjective(3*x + 4*y, GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2*y <= 14, "c1")
model.addConstr(3*x - y >= 0, "c2")
model.addConstr(x - y <= 2, "c3")

# Optimize the model
model.optimize()

# Print the results
for v in model.getVars():
    print(f'{v.varName}: {v.x}')
print(f'Objective: {model.objVal}')

"""
Created on Wed Oct 23 20:26:36 2024

@author: Daniel
"""

