# Bench mark function 16
# Six-Hump Camel-Back Function
# HW dimension: 2
# Min = -1.031628453489877
# Range [-5,5]
# Reference: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/23-six-hump-camel-back-function

import math
import numpy as np

l_bound = -5
u_bound = 5
dim = 2
opt = -1.031628453489877

def func(X):
    t1 = 4 * X[0]**2
    t2 = -1 * 2.1 * X[0]**4
    t3 = X[0]**6 / 3
    t4 = np.prod(X)
    t5 = -1 * 4 * X[1]**2
    t6 = 4 * X[1]**4
    result = t1 + t2 + t3 + t4 + t5 + t6
    return result

if __name__ == '__main__':
    X = np.arange(2)  # Dimension must be 2, in [-5 to 5]
    print(X)
    result = func(X)
    print(result)