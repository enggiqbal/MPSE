import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

import gd, mds

##########

# Code for dealing with the multiview-MDS problem w/ varying node positions and
# projection matrices. The nodes are assumed to live in R3 and the projections
# are assumed to be of rank 2. For some functions, it is assumed that there are
# exactly three perspectives under consideration.

##########

##### Helper functions #####

### Radially Symmetric X data ###

def random_disk(n,feedback=False):
    r = np.random.rand(n)
    X0 = np.random.randn(n,2)
    X = (X0.T / np.linalg.norm(X0,axis=1) * np.sqrt(r)).T
    if feedback is True:
        plt.figure()
        plt.plot(X[:,0],X[:,1],'o')
        plt.axis('equal')
        plt.title('Points in disk uniformly sampled')
        plt.show()
    return X

def random_ssphere(n=1000,feedback=False):
    r = np.random.rand(n)
    X0 = np.random.randn(n,3)
    X = (X0.T / np.linalg.norm(X0,axis=1) * r**(1/3)).T
    if feedback is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2],'o')
        ax.set_aspect('equal')
        plt.title('Points in solid sphere uniformly sampled')
        plt.show()
    return X

### Orthogonal 3x2 matrices ###

def standard_orthogonal_matrices():
    """\
    Returns list with the three orthogonal matrices corresponding to the
    cannocial rank 2 projections in R3.
    """
    Q_list = []
    for k in range(3):
        Q = np.delete(np.identity(3),k,1)
        Q_list += [Q]
    return Q_list

def random_orthogonal_matrices(number):
    """\
    Returns a list contaning specified number of orthogonal 3x2 matrices.
    """
    Q_list= []
    for i in range(number):
        Q = np.random.randn(3,2)
        Q = mds.projection_to_orthogonal(Q)
        Q_list += [Q]
    return Q_list

def normal_vectors(Qs):
    """\
    Takes a list of orthogonal 3x2 matrices and returns a list with the
    corresponding normal vectors.
    """
    vs = []
    for i in range(len(Qs)):
        v = np.cross(Qs[i][:,0],Qs[i][:,1])
        vs += [v]
    return vs

def orthogonal_distances(Qs1,Qs2):
    dist = []
    vs1 = normal_vectors(Qs1)
    vs2 = normal_vectors(Qs2)
    for i in range(len(Qs1)):
        dist = [np.arccos(np.dot(vs1[i],vs2[i]))]
    return dist

def compute_projections(Qs):
    """\
    Takes a list of orthogonal 3x2 matrices and returns a list with the
    corresponding 3x3 projection matrices.
    """
    Ps = []
    for i in range(len(Qs)):
        Ps += [Qs[i] @ Qs[i].T]
    return Ps
        
### Distance matrices ###

def combine_distances(Ds):
    """\
    Combine list of distance matrices into one by assuming that the 
    corresponding distance matrices are orthogonal

    --- arguments ---
    Ds = list of three distance matrices (one per perspective)

    Note : It is assumed that Ds contains 3 distance matrices. The formula is
    only correct for data formed by rank 2 projections in R3 that are orthogonal
    to each other.
    """
    return ((Ds[0]**2+Ds[1]**2+Ds[2]**2)/2)**(1/2)
    
def compute_distance_matrices(X,Qs):
    """\
    Return list of distance matrices produced by distances of X after projection
    by transformations in the list Qs

    --- arguments ---
    X = positions (n x p)
    Qs = list of orthogonal matrices
    """
    Ds = []
    for i in range(len(Qs)):
        Q = Qs[i]
        temp = X @ (Q @ Q.T)
        Ds += [distance_matrix(temp,temp)]
    return Ds

##### Optimization algorithms #####

### Learning rate functions ###

def set_init_rates(n):
    """\
    Approximate good set of rates for solving for X0 and Q0
    """
    return [0.1/n,0.001/n**2]

def set_rates(n):
    """\
    Approximate good set of rates for solving for X and Q
    
    --- arguments ---
    n = number of points
    """
    return [0.01/n,0.001/n**2]

### Gradient descent algorithms ###

def find_X0(Ds,X0=None,feedback=False,**kwargs):
    """\
    Solution to MDS problem using combined distance matrix. The list of distance
    matrices Ds is combined into one, and the node positions in R3 are computed
    using regular MDS, starting with initial configuration X0.

    --- arguments ---
    Ds = list of distance matrices (k x n x n)
    X0 = initial positions, organized by row (n x p)

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion

    Note : It is assumed that Ds contains 3 distance matrices. The formula is
    only correct for data formed by rank 2 projections in R3 that are orthogonal
    to each other.
    """
    if feedback is True:
        print("\nBegin find_X0():")

    n = len(Ds[0])
    params = {
        'rate' : None,
        'max_iters': 1000
    }
    params.update(kwargs)
    if params['rate'] is None:
        params['rate'] = set_init_rates(n)[0]
        
    D = combine_distances(Ds)
    if X0 is None:
        X0 = random_ssphere(n)
        
    X = mds.MDS_descent(D,X0,feedback=feedback,**params)

    if feedback is True:
        print("\nEnd find_X0()")
        
    return X

def find_Q0(Ds,X,feedback=False,**kwargs):
    """\
    Solution to MDS problem using combined distance matrix. The list of distance
    matrices Ds is combined into one, and the node positions in R3 are computed
    using regular MDS, starting with initial configuration X0.

    --- arguments ---
    X = positions (n x p)
    Ds = list of distance matrices (k x n x n)

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion

    Note : It is assumed that Ds contains 3 distance matrices. The formula is
    only correct for data formed by rank 2 projections in R3 that are orthogonal
    to each other.
    """
    if feedback is True:
        print("\nBegin find_Q0():")

    n = len(Ds[0])
    params = {
        'rate' : None,
        'max_iters': 1000
    }
    params.update(kwargs)
    if params['rate'] is None:
        params['rate'] = set_init_rates(n)[1]

    Q0s = standard_orthogonal_matrices()
    Qs = []
    for i in range(3):
        Q = mds.MDSq_Qdescent(X,Q0s[i],Ds[i],feedback=feedback,**params)
        Qs += [Q]

    if feedback is True:
        stress0 = mds.mMDSq_stress(X,Q0s,Ds)
        print(f"\n initial stress = {stress0:.2e}")
        stress1 = mds.mMDSq_stress(X,Qs,Ds)
        print(f" final stress = {stress1:.2e}")
        print("\nEnd find_Q0().")
        
    return Qs

def find_XQ(Ds,X0,Q0s,feedback=False,**kwargs):
    if feedback is True:
        print("\nBegin find_XQ():")

    n = len(Ds[0])
    params = {
        'rates' : None,
        'loops': 20,
        'max_iters': [100,50]
    }
    params.update(kwargs)
    if params['rates'] is None:
        params['rates'] = set_rates(n)

    x = X0.copy()
    qs = Q0s.copy()
    stress = np.zeros(params['loops'])
    for i in range(params['loops']):
        x0 = x.copy(); qs0 = qs.copy()
        x = mds.mMDSq_Xdescent(x0,qs0,Ds,feedback=feedback,
                               rate=params['rates'][0],
                               max_iters=params['max_iters'][0])
        for j in range(3):
            qs[j] = mds.MDSq_Qdescent(x,qs0[j],Ds[j],feedback=feedback,
                                      rate=params['rates'][1],
                                      max_iters=params['max_iters'][1])
        stress[i] = mds.mMDSq_stress(x,qs,Ds)
        if feedback is True:
            print(f"\n multiview stress = {stress[i]:.2e}")
    X = x; Qs = qs
    
    if feedback is True:
        stress0 = mds.mMDSq_stress(X0,Q0s,Ds)
        print(f"\n initial stress = {stress0:.2e}")
        stress1 = mds.mMDSq_stress(X,Qs,Ds)
        print(f" final stress = {stress1:.2e}")
        print("\nEnd find_XQ().")

        return X,Qs,stress

def optimal_XQ(Ds,X0=None,Q0s=None,feedback=False,**kwargs):
    if feedback is True:
        print("\nBegin optimal_XQ():")

    n = len(Ds[0])
    params = {
        'rates' : None,
        'loops': 20,
        'max_iters': [100,50],
        'init_rates' : None,
        'init_max_iters': [2000,2000],
    }
    params.update(kwargs)
    if params['rates'] is None:
        params['rates'] = set_rates(n)
    if params['init_rates'] is None:
        params['init_rates'] = set_init_rates(n)

    X0 = find_X0(Ds,X0=X0,feedback=feedback,
                 rate=params['init_rates'][0],
                 max_iters=params['init_max_iters'][0])
    Q0s = find_Q0(Ds,X0,feedback=feedback,rate=params['init_rates'][1],
                  max_iters=params['init_max_iters'][1])
    stress0 = mds.mMDSq_stress(X0,Q0s,Ds)
    X,Qs,stress = find_XQ(Ds,X0,Q0s,feedback=feedback,**params)
    stress = [stress0]+stress

    if feedback is True:
        print("\nEnd optimal_XQ().")
        
    return X,Qs,stress,X0,Q0s

##### Node Assignment #####

def find_Xp(X0,D0_list,Q_list,cycles=10,feedback=False,**kwargs):
    """\
    Find perturbation of labels of each D to improve fit

    --- arguments ---
    X = positions (n x p)
    Qs = list of orthogonal matrices
    Ds = list of distance matrices (k x n x n)
    """
    if feedback is True:
        print("\nBegin find_Xp():")

    K = len(Ds)
    n = len(Ds[0])

    D_list = D0_list.copy()
    for i in range(cycles):
        d_list = mds.dmatrices(X,Q_list=Qlist)
        sigma_list = assign.reassign(d_list,D_list,**kwargs)
        D_list = update_dmatrices(D_list,sigma_list)
        X = mds.mMDSq_Xdescent(X,Q_list,D_list)

    return X,D_list,overall_perm
