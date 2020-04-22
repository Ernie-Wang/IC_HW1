# Bench mark function 12
# Generalized Penalized Function No.01
# HW dimension: 30
# Min = 0
# Range [-50,50]
# Reference: https://al-roomi.org/benchmarks/unconstrained/n-dimensions/172-generalized-penalized-function-no-1

import math
import numpy as np

name = "F12"
l_bound = -50
u_bound = 50
dim = 30
opt = 0

def u(u_X, a, k, m):
    X = u_X.copy()
    for index in range(X.size):
        element = X[index]
        if element > a:
            X[index] = k * (element - a)**m
        elif element < -1 * a:
            X[index] = k * (-1 * element - a)**m
        else:
            X[index] = 0
    return X

def func(X):
    Y = 1 + (X+1)/4
    f_term = 10 * math.sin(math.pi*Y[0])**2
    m_1 = (Y[0:-1] - 1)**2
    m_2 = 1 + 10 * np.sin(math.pi*Y[1:])**2
    m_term = np.sum(np.multiply(m_1, m_2))
    l_term = (Y[-1]-1)**2
    front = math.pi * (f_term + m_term + l_term) / X.size
    end = np.sum(u(X, 10, 100, 4))
    result = front + end
    return result

if __name__ == '__main__':
    X = np.arange(10)
    X = X * 0.3
    print(X)
    result = func(X)
    print(result)