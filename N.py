# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:52:25 2024

@author: Daniel
"""

import numpy as np

##set seed
np.random.seed(37)

## Coordinates between 0 and 500 Kilometers
coordinates = np.random.uniform(0,500, size =(100,2) )

## coordinates hights between 0 and 300 in Meters
h = np.random.uniform(0,300, size = (100,1))

## demand between 100 and 1000
q = np.random.uniform(100,1000, size =(100,1) )

## dropoff windows between 0 and 720 minutes
a = np.zeros((100,1))
b = np.full((100,1), 720)

## merge into 1 array
N = np.hstack((coordinates, h, q, a, b))

##create Depot
N0 = np.array([[250, 50, 150, np.nan, np.nan, np.nan]])

## merge
N = np.vstack((N0, N))
## store data for use in other programs
np.save('N.npy', N)
print(N)