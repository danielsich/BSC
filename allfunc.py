from Dist import dijin
from radial import ang, aij, b
from speedlevelarchs import levels, tj0


import numpy as np
N = np.load('N.npy')
def test(a,N):
    dijin(a, N)
    return dijin
a = test(30,N)
print(a)