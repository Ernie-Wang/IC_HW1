# Bench mark function 7
# Quartic Function
# HW dimension: 30
# Min = 0+random noise when X={0,0,0,....0}
# Range [-30,30]
# Reference: http://benchmarkfcns.xyz/benchmarkfcns/quarticfcn.html

import math
import random
import numpy as np

def func(X):
    index = np.arange(1, 1+X.size)
    f_term = np.sum(np.multiply(index, X**4))
    tmp = f_term + random.random()
    return tmp

if __name__ == '__main__':
    X = np.array([1, 1, 1, 1, 1])
    # X = np.arange(10)
    # X = X * 0.3
    print(X)
    result = func(X)
    print(result)