# Bench mark function 24
# Ackley Function
# HW dimension: 30
# Min = 0
# Range [-32.768,32.768]
# Reference: http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html

import math
import numpy as np

name = "F24"
l_bound = -32.768
u_bound = 32.768
dim = 30
opt = 0

def func(X):
    n = X.size
    term_1 = 20
    term_2 = np.exp(1)
    term_3 = -20*np.exp(-0.2*(np.sum(X**2)/n)**0.5)
    term_4 = -1*np.exp(np.sum(np.cos(2*math.pi*X))/n)
    result = term_1 + term_2 + term_3 + term_4
    return result

if __name__ == '__main__':
    X = np.arange(2)  # Dimension must be 2, in [-5 to 5]
    print(X)
    result = func(X)
    print(result)