# Bench mark function 22
# {Unknown Name}
# HW dimension: 4
# Min = -10.4028
# Range [0,10]
# Reference: 

import math
import numpy as np

l_bound = 0
u_bound = 10
dim = 4
opt = -10.4028
table = np.array([[4,4,4,4],
                    [1,1,1,1],
                    [8,8,8,8],
                    [6,6,6,6],
                    [3,7,3,7],
                    [2,9,2,9],
                    [5,5,3,3],
                    [8,1,8,1],
                    [6,2,6,2],
                    [7,3.6,7,3.6]])
c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 
                0.6, 0.3, 0.7, 0.5, 0.5])

def func(X):
    term_0 = X-table[0:7]
    term_1 =  np.einsum('ij,ji->i', term_0, term_0.T) + c[0:7]
    result = -1 * np.sum(1/term_1)
    return result

if __name__ == '__main__':
    X = np.arange(4)  # Dimension must be 2, in [-5 to 5]
    print(X)
    result = func(X)
    print(result)