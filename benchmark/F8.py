# Bench mark function 8
# Generalized Schwefel's Function No.2.26
# HW dimension: 30
# Min = -418.982887272433799807913601398 * n
# Range [-500,500]
# Reference: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/176-generalized-schwefel-s-problem-2-26

import math
import numpy as np

l_bound = -500
u_bound = 500
dim = 30
opt = -418.982887272433799807913601398 * dim

def func(X):
    sin_term = np.sin(np.absolute(X)**0.5)
    result = np.sum(-1*np.multiply(X, sin_term))
    return result

if __name__ == '__main__':
    X = np.arange(30)
    X = X * 0.3
    print(X)
    result = func(X)
    print(result)