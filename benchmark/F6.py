import math
import numpy as np

def F_6(X):
    sum = 0
    for x in X:
        sum = sum + math.pow( math.floor(x+0.5), 2)
    return sum

if __name__ == '__main__':
    # X = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
    X = np.arange(10)
    X = X * 0.3
    print(X)
    sum = F_6(X)
    print(sum)