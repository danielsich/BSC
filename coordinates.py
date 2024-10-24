# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:52:25 2024

@author: Daniel
"""

import numpy as np

## Coordinates between 0 and 100
coordinates = np.random.uniform(0,100, size =(100,2) )

## demand between 100 and 1000
q = np.random.uniform(100,1000, size =(100,1) )

## dropoff windows
a = np.zeros((100,1))
b = np.full((100,1), 720)

## merge into 1 array
data = np.hstack((coordinates,q,a,b))
print(data)