import numpy as np 
import copy
import random
import matplotlib.pyplot as plt

length = 5

a1_col = np.zeros(length)
a2_col = np.zeros(length)

cols=[a1_col,a2_col]

Ta1 = np.diag(np.ones(length-1),k=1)
Ta1[-1][-1]=1
Ta2 = np.zeros((length,length-1))
Ta2 = np.hstack((np.ones(length).reshape((length,1)),Ta2))

Tas = [Ta1,Ta2]

ra1 = np.zeros(length)
ra1[-1] =1
ra2     = np.zeros(length)
ra2[0]  = 0.2

rs = [ra1,ra2]
alpha = 0.9
beta = 0.9

def eps_greedy(cols,x,eps=0.1):
    if random.random()<eps:
        return random.randint(0,1)
    else:
        return np.argmax((cols[0][x],cols[1][x]))

for i in range(100):
    old_cols = copy.deepcopy(cols)
    for x in range(length):
        svector = np.zeros(length)
        svector[x] = 1
        a = eps_greedy(cols,x)
        r = np.dot(rs[a],svector)
        x_prime = np.dot(svector,Tas[a])
        max_q = max(np.dot(x_prime,old_cols[0]),np.dot(x_prime,old_cols[1]))
        cols[a][x] += alpha * (r + beta*max_q - cols[a][x])
    print(cols)

