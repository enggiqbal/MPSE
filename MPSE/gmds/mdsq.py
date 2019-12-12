import numpy as np
import matplotlib.pyplot as plt

import gd, mds

## Orthogonal projections ##

def nearest_orthogonal(Q0):
    """\
    Returns nearest orthogonal matrix to Q0.

    --- arguments ---
    Q0 = any p x k matrix

    --- notes ---
    The nearest orthogonal matrix Q to Q0 is given by UV^T, where Q0=UsV^T is
    the SVD decomposition of Q0.
    """
    U,s,Vh = np.linalg.svd(Q0, full_matrices=False)
    return U @ Vh

def random_orthogonal(p,k):
    """\
    Returns a random orthogonal p x k matrix.

    --- arguments ---
    p, k = dimension of random matrix
    """
    Q0 = np.random.randn(p,k)
    Q = nearest_orthogonal(Q0)
    return Q

def grassmanian_distance(Q1,Q2):
    """\
    Grassmanian distance between two orthogonal matrices of the same shape.
    """
    return np.linalg.norm(Q1 @ Q1.T - Q2 @ Q2.T,ord=2)
    
## Projection and distance matrix ##

def projected_positions(X,Q):
    """\
    Returns positions of nodes as given by XQQ^T. If Q is a list, then it a list
    of position matrices is given, one for each orthogonal matrix in the list.
    
    --- arguments ---
    X = positions (n x p)
    Q = rank k orthogonal matrix (p x k) or list
    """
    return X @ (Q @ Q.T)

def dmatrix(X,Q):
    """\
    Return distance matrix of nodes with positions given by XQQ^T.
    """
    Y = projected_positions(X,Q)
    d = mds.dmatrix(Y)
    return d

### MDSq optimization ###

def stress(X,Q,D):
    """\
    Returns MDS stress between distance matrix of XQQ^T and D.

    --- arguments ---
    X = node positions
    Q = orthogonal matrix or list of orthogonal matrices
    D = observed distance matrix or list
    """
    Y = projected_positions(X,Q)
    s2 = mds.stress(Y,D)
    return s2

def Xgradient(X,Q,D):
    """\
    Gradient of s2(X*P.T;D) w.r.t. X

    X : data positions (n x p)
    Q : orthogonal matrix (p x x)
    D : target distances (n x n)
    """
    P = Q @ Q.T
    return mds.gradient(X @ P.T, D) @ P

def Qgradient(X,Q,D):
    """\
    Gradient of MDS stress w.r.t. Q.
    """
    Y = projected_positions(X,Q)
    temp2 = mds.gradient(Y,D).T @ X
    gradient = (temp2 + temp2.T) @ Q
    return gradient

def Qdescent(X,Q0,D,feedback=False,plot=False,**kwargs):
    """\
    MDSq Q optimization

    --- arguments ---
    X = node positions
    Q0 = initial orthogonal matrix
    D = target distances
    feedback = set to True to print feedback
    plot = set to True to return stress plot
    """
    if feedback is True:
        print("\nmdsq.Qdescent():")
        print(f" initial stress = {stress(X,Q0,D):.2e}")
    if plot is True:
        kwargs['cost_history'] = True
        kwargs['step_history'] = True
        
    df = lambda Q: Qgradient(X,Q,D)
    f = lambda Q: stress(X,Q,D)
    p = lambda Q: nearest_orthogonal(Q)

    results = gd.gradient_descent(Q0,df,projection=p,f=f,plot_history=True,
                                  **kwargs)
    
    if feedback is True:
        Q = results['output']
        print(f" final stress = {stress(X,Q,D):.2e}")
    if plot is True:
        plt.show()
        
    return results

##### Tests #####

def example1():
    """\
    Example of MDSq algorithm for exact random points in plane

    The positions X in R3 are random and fixed. 
    The orthogonal matrix Q is projection onto the yz plane.
    The initial orthogonal matrix Q0 is choosen randomly.
    """
    print('\n##### TEST #####')
    print('mdsq.example1():')
    print('Orthogonal projection projecting onto the yz plane is recovered '\
          'from points \nin cube and distance matrix after projection.')
    n = 10 #number of points
    X = np.random.rand(n,3) #positions'
    Q = np.array([[0,1,0],[0,0,1]]).T #orthogonal matrix
    D = dmatrix(X,Q) #distances of new positions

    Q0 = random_orthogonal(3,2)
    results = Qdescent(X,Q0,D,feedback=True,plot=True,rate=.01)
    Qf = results['output']
    
    print('\n  true orthogonal matrix:')
    print(Q)
    print('  initial orthogonal matrix:')
    print(Q0)
    print(f'  grassmanian distance to original: {grassmanian_distance(Q,Q0)}')
    print('  final orthogonal matrix:')
    print(Qf)
    print(f'  grassmanian distance to original: {grassmanian_distance(Q,Qf)}')
