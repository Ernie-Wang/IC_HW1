# Bench mark function 11
# Griewank Function
# HW dimension: 30
# Min = 0 when X={0,0,0,....0}
# Range [-600,600]
# Reference: http://benchmarkfcns.xyz/benchmarkfcns/griewankfcn.html

import math
import numpy as np

l_bound = -600
u_bound = 600

def func(X):
    agg = np.sum(X**2)/4000
    index = np.arange(1, 1+X.size)
    cos_term = np.cos(np.divide(X,index**0.5))
    multi = np.prod(cos_term)
    result = agg - multi + 1
    return result

if __name__ == '__main__':
    X = np.arange(30)
    X = X * 0.3
    print(X)
    result = func(X)
    print(result)