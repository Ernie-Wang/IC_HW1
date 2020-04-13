# Bench mark function 8
# Generalized Schwefel's Function No.2.26
# Min = −418.982887272433799807913601398 * n
# Reference: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/176-generalized-schwefel-s-problem-2-26

import math
import numpy as np

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