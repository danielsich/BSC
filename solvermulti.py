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
import time

load_dotenv()

Distall = np.load('Distall.npy')
Nstart = np.load('N.npy')

## set seed
np.random.seed(37)

## set up the possible combinations
def tupls (a):
    arcs = [(i, j) for i in range(a+1) for j in range(a+1) if i != j]
    return np.array(arcs)

# string relevant customers
def relevantcusta(a,N):
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

#define relvant Distnces
def relevantdistances(relevantcusta,Distall):
    size = len(relevantcusta)
    Dist = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            Dist[i,j] = Distall[relevantcusta[i],relevantcusta[j]]

    return Dist

##define averagespeed for the relevant speed levels
def levels(low, high, level):
    diff = high - low
    ranges = diff/level
    lvl = [(low + i * ranges + 0.5 * ranges) for i in range(level)]
    return np.array(lvl)

# time2depot
def tj00(levels, Dist):
    assert Dist.ndim == 2
    assert levels.ndim == 1
    trvlt = np.zeros((Dist.shape[0], levels.shape[0]))
    for i in range(Dist.shape[0]):
        for j in range(len(levels)):
            trvlt[i, j] = Dist[i, 0] / levels[j]
    return trvlt

# angle calculation
def ang(N):
    n = N.shape[0]
    ang = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i != j):
                dot = np.dot(N[i], N[j])
                magi = np.linalg.norm(N[i])
                magj = np.linalg.norm(N[j])
                cos_theta = dot / (magi * magj)
                cos_theta = np.clip(cos_theta,-1.0, 1.0)
                ang[i, j] = np.arccos(cos_theta)
    return ang

# constant calc arch
def aij(a, Cr, angl):
    g = 9.81  # gravity
    a_ij = np.zeros_like(angl)

    for i in range(angl.shape[0]):
        for j in range(angl.shape[1]):
            a_ij[i, j] = a + g * np.sin(angl[i, j]) + g * Cr * np.cos(angl[i, j])

    return a_ij

#vehicle constant
def b(cd,A,p):
    return 0.5 * cd * A * p

#totaldistancecalc
def calculate_total_distance(xVar, Dist):
    total_distance = 0
    for (i, j) in xVar.keys():
        if xVar[i, j] == 1:
            total_distance += Dist[i, j]
    return total_distance

def calculate_average_speed(xVar, Dist, speed, lvl):
    total_distance = 0
    total_time = 0
    for (i, j) in xVar.keys():
        if xVar[i, j] == 1:
            total_distance += Dist[i, j]
            for r in range(len(lvl)):
                if speed[i, j, r] > 0.5:
                    total_time += Dist[i, j] / lvl[r]
                    break
    if total_time == 0:
        return 0
    return total_distance / total_time

# prp cost calc
def calculate_total_costs(xVar, f, Dist, a_ij, cfe, W, betaa, lvl):
    total_cost = 0
    for (i, j) in xVar.keys():
        if xVar[i, j] == 1:
            total_cost += cfe * a_ij[i, j] * Dist[i, j] * W
            total_cost += cfe * a_ij[i, j] * f[i, j] * Dist[i, j]
            for r in range(len(lvl)):
                if z[i, j, r] == 1:
                    total_cost += cfe * Dist[i, j] * betaa * (lvl[r] ** 2)
    return total_cost


for a in range(5):
    #set parameters
    inp = relevantcusta(10,Nstart)
    relN = relevantcustomers(inp,Nstart)
    Dist = relevantdistances(inp,Distall)
    Archs = tupls(10)
    lvl = levels((40/60), (100/60), 60)
    tj0 =tj00(lvl,Dist)
    angl = ang(relN[:, :3])
    a_ij = aij(0, 0.01, angl)
    betaa = b(0.7, 5, 1.2041)

    ## customers
    N0d = relN[1:]  # information all customers

    N = np.arange(relN.shape[0])  # All edges
    N0 = N[1:]  # alle customer edges
    Nq = N.shape[0]  # number edges
    N0q = N0.shape[0]  # number customers
    qi = relN[:, 3]  # Demand Customer
    ti = 1  # service time
    ai = relN[:, 4]  # earliest time
    bi = relN[:, 5]  # latest time

    # print(qi)
    ##Vehicles
    m = 9  # amount
    K = np.arange(m)
    Q = 3500  # Capacity
    W = 3500  # curb weight
    p = 1
    cfe = 2  # cost for fuel and emissions
    BIGM = 999999999  ##bigM

    options = {
        # configure()
        "WLSACCESSID": os.getenv("WLSACCESSID"),
        "WLSSECRET": os.getenv("WLSSECRET"),
        "LICENSEID": 2503389,
    }

    # Before optimization
    start_time = time.time()
    # start Model
    prp = gp.Model(env=gp.Env(params=options))
    x = {}  ## binary decison variable 1 if the arch is driven
    for i in range(Nq):
        for j in range(Nq):
            x[i, j] = prp.addVar(vtype=GRB.BINARY)

    z = {}  ## available speedlevels for each arch
    for i in range(Nq):
        for j in range(Nq):
            for r in range(lvl.shape[0]):
                z[i, j, r] = prp.addVar(vtype=GRB.BINARY)

    f = {}  ## amout of product flowing through each arch
    for i in range(Nq):
        for j in range(Nq):
            f[i, j] = prp.addVar(vtype=GRB.CONTINUOUS)

    y = {}  ## time at wich node i is visited
    for i in range(Nq):
        y[i] = prp.addVar(vtype=GRB.CONTINUOUS)

    s = {}
    for i in range(Nq):
        s[i] = prp.addVar(vtype=GRB.CONTINUOUS)

    zfpd = quicksum(x[i, j] * Dist[i, j] for i in N for j in N)
    zfpl = quicksum(a_ij[i, j] * Dist[i, j] * W * x[i, j] for i in N for j in N) + quicksum(
        a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N)
    zfpe = quicksum(a_ij[i, j] * Dist[i, j] * W * x[i, j] for i in N for j in N) + quicksum(
        a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N) + quicksum(
        Dist[i, j] * betaa * (quicksum((lvl[r] ** 2) * z[i, j, r] for r in range(lvl.shape[0]))) for i in N for j in N)
    zfprp = quicksum(cfe * a_ij[i, j] * Dist[i, j] * W * x[i, j] for i in N for j in N) + quicksum(
        cfe * a_ij[i, j] * f[i, j] * Dist[i, j] for i in N for j in N) + quicksum(
        cfe * Dist[i, j] * betaa * (quicksum((lvl[r] ** 2) * z[i, j, r] for r in range(lvl.shape[0]))) for i in N for j
        in N) + quicksum(p * s[j] * x[j, 0] for j in N0)
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
        prp.addConstr(qi[j] * x[i, j] <= f[i, j], name=f"con14_low_{i}_{j}")
        prp.addConstr(f[i, j] <= (Q - qi[i]) * x[i, j], name=f"con14_high_{i}_{j}")

    for i in N:
        for j in N0:
            if i != j:
                prp.addConstr(y[i] - y[j] + ti + quicksum(
                    (Dist[i, j] / lvl[r]) * z[i, j, r] for r in range(lvl.shape[0])) <= BIGM * (1 - x[i, j]),
                              name=f"con_15_{i}_{j}")

    for i in N0:
        prp.addConstr(y[i] >= ai[i], name=f"con_16_low_{i}")

    for i in N0:
        prp.addConstr(y[i] <= bi[i], name=f"con16_high_{i}")

    for i in N0:
        prp.addConstr(
            y[i] + ti - s[i] + quicksum((Dist[i, 0] / lvl[r]) * z[i, 0, r] for r in range(lvl.shape[0])) <= BIGM * (
                        1 - x[i, 0]), name=f"con_17_{i}")

    for i, j in Archs:
        prp.addConstr(quicksum(z[i, j, r] for r in range(lvl.shape[0])) == x[i, j], name=f"con_18_{i}_{j}")

    for j in N0:
        if j != 0:
            prp.addConstr((y[j] + ti + quicksum(tj0[j, r] * z[j, 0, r] for r in range(lvl.shape[0]))) * x[j, 0] == s[j],
                          name=f"constr_22{j}")

    for i, j in Archs:
        prp.addConstr(f[i, j] >= 0, name=f"con_20_{i}_{j}")

    for i in N0:
        for j in N0:
            if i != j:
                prp.addConstr(x[i, j] + x[j, i] <= 1, name='subtourbreaking')
    ##set params
    prp.setParam('TimeLimit', 10000)
    prp.setParam('OutputFlag', 0)

    prp.update()
    prp.optimize()

    # After optimization
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to calculate the Gurobi model: {elapsed_time} seconds")
    xVar = prp.getAttr('x', x)
    total_distance = calculate_total_distance(xVar, Dist)
    print(f"Total Distance Traveled: {total_distance}")
    speed = prp.getAttr('x', z)
    average_speed = calculate_average_speed(xVar, Dist, speed, lvl)
    print(f"Average Speed of Vehicles: {average_speed:.2f}")
    total_costs = calculate_total_costs(xVar, f, Dist, a_ij, cfe, W, betaa, lvl)
    print(f"Total Costs: {total_costs}")



