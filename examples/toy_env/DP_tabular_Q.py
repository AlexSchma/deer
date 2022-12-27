import numpy as np 
import copy

a1_col = np.zeros(5)
a2_col = np.zeros(5)

cols=[a1_col,a2_col]

Ta1 = np.diag(np.ones(4),k=1)
Ta1[-1][-1]=1
Ta2 = np.zeros((5,4))
Ta2 = np.hstack((np.ones(5).reshape((5,1)),Ta2))

Tas = [Ta1,Ta2]

ra1 = np.zeros(5)
ra1[-1] =1
ra2     = np.zeros(5)
ra2[0]  = 0.2

rs = [ra1,ra2]


for i in range(2000):
    old_cols = copy.deepcopy(cols)
    for x in range(5):
        svector = np.zeros(5)
        svector[x] = 1
        for a in range(2):
            r = np.dot(rs[a],svector)
            x_prime = np.dot(svector,Tas[a])
            max_q = max(np.dot(x_prime,old_cols[0]),np.dot(x_prime,old_cols[1]))
            cols[a][x]=r+0.9*max_q
    print(cols)
  