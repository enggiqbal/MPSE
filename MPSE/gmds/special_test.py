import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix

import mds, special

### Initialization ###

def initialization_test0(n=10,randomQs=False,rates=[0.01,0.01]):
    print("\n##########")
    print('\nBegin initialization_test0():')
    
    X = special.random_ssphere(n) #true positions
    if randomQs is True:
        Qs = special.random_orthogonal_matrices(3)
    else:
        Qs = special.standard_orthogonal_matrices()
    Ds = special.compute_distance_matrices(X,Qs)

    X0 = special.find_X0(Ds,feedback=True,rate=rates[0])
    Q0s = special.find_Q0(Ds,X0,feedback=True,rate=rates[1])

    print("\nEnd initialization_test().")
    print("\n##########")

def optimization_test0(n=10,rates=[0.01,0.01]):
    print("\n##########")
    print('\nBegin optimization_test0():')
    
    X = special.random_ssphere(n) #true positions
    Qs = special.random_orthogonal_matrices(3)
    Ds = special.compute_distance_matrices(X,Qs)

    X0 = special.find_X0(Ds,feedback=True,rate=rates[0])
    Q0s = special.find_Q0(Ds,X0,feedback=True,rate=rates[1])
    X1,Qs1,stress = special.find_XQ(Ds,X0,Q0s,feedback=True,rates=rates)

    print("\nEnd optimization_test0().")
    print("\n##########")
