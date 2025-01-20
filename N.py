# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:52:25 2024

@author: Daniel
"""

import numpy as np

##set seed
np.random.seed(37)

## Coordinates between 0 and 500 Kilometers in Meters
coordinates = np.random.uniform(0,500000, size =(500,2) )

## coordinates hights between 0 and 300 in Meters
h = np.random.uniform(0,300, size = (500,1))

## demand between 100 and 1000
q = np.random.uniform(400,600, size =(500,1) )

## dropoff windows between 0 and 720 minutes
a = np.zeros((500,1))
b = np.full((500,1), 43200)

## merge into 1 array
N = np.hstack((coordinates, h, q, a, b))

##create Depot
N0 = np.array([[250000, 250000, 150, 0, 0, 600]])

## merge
N = np.vstack((N0, N))
## store data for use in other programs
np.save('N.npy', N)
print(N)