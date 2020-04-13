import math
import numpy as np

def func(X):
    tmp = 0
    for x in X:
        tmp = tmp + math.pow( math.floor(x+0.5), 2)
    return tmp

if __name__ == '__main__':
    # X = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
    X = np.arange(10)
    X = X * 0.3
    print(X)
    result = func(X)
    print(result)