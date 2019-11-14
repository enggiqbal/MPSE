import numpy as np

def input(m,n,seed):
    M=100*np.random.randn(m, n) + 1
    return M
print(input(5,5,124))
