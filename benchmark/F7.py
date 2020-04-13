import math
import random
import numpy as np

def func(X):
    tmp = 0
    for i, x in enumerate(X):
        rand = 1
        while rand == 1:
            rand = random.random()
        tmp = tmp + math.pow(x, 4) * (i + 1) + rand
    return tmp

if __name__ == '__main__':
    X = np.array([1, 1, 1, 1, 1])
    # X = np.arange(10)
    # X = X * 0.3
    print(X)
    result = func(X)
    print(result)