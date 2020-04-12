import math
import random
import numpy as np

def F_7(X):
    sum = 0
    for i, x in enumerate(X):
        rand = 1
        while rand == 1:
            rand = random.random()
        sum = sum + math.pow(x, 4) * (i + 1) + rand
    return sum

if __name__ == '__main__':
    X = np.array([1, 1, 1, 1, 1])
    # X = np.arange(10)
    # X = X * 0.3
    print(X)
    sum = F_7(X)
    print(sum)