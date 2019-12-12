import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

import gd, mds, mdsq

### Orthogonal projections ###

def nearest_orthogonal(Q0s):
    """\
    Returns list with nearest orthogonal matrix for each matrix in list
    
    --- arguments ---
    Q0s = list of matrices
    """
    Qs = []
    for i in range(len(Q0s)):
        Qs += [mdsq.nearest_orthogonal(Q0s[i])]
    return Qs

def random_orthogonal(p,k,number=1):
    """\
    Returns a list of random orthogonal p by k matrices.
    """
    Q0s = []
    for i in range(number):
        Q0s += [np.random.randn(p,k)]
    Qs = nearest_orthogonal(Q0s)
    return Qs

def standard_orthogonal(p,k):
    """\
    Returns list with standard orthogonal p by k matrices
    """
    import itertools
    assert p >= k
    Qs = []
    for comb in itertools.combinations(range(p),k):
        Qs += [np.identity(p)[:,comb]]
    return Qs

def cylinder(num=3):
    import math
    Qs = []
    for k in range(num):
        theta = math.pi/num*k
        Q = np.array([[math.cos(theta),0],[math.sin(theta),0],[0,1]])
        Qs.append(Q)
    return Qs

def grassmanian_distance(Q1s,Q2s):
    dist = []
    for i in range(len(Q1s)):
        dist += [mdsq.grassmanian_distance(Q1s[i],Q2s[i])]
    return dist

def projected_positions(X,Qs):
    """\
    Returns list of node positions after orthogonal projections in the list.
    
    --- arguments ---
    X = node positions
    Qs = list of orthogonal matrix (p x k)
    """
    Y = []
    for i in range(len(Qs)):
        Y.append(mdsq.projected_positions(X,Qs[i]))
    return Y

def dmatrices(X,Qs):
    """\
    Return list with distance matrices of nodes after orthogonal projectiosn in 
    the list.
    """
    d = []
    for i in range(len(Qs)):
        d.append(mdsq.dmatrix(X,Qs[i]))
    return d

### multiview MDS optimizers ###

def stress(X,Qs,Ds):
    """\
    Returns MDS stress between distance matrix of XQQ^T and D.

    --- arguments ---
    X = node positions
    Q = orthogonal matrix or list of orthogonal matrices
    D = observed distance matrix or list
    """
    s2 = 0
    for i in range(len(Qs)):
        s2 += mdsq.stress(X,Qs[i],Ds[i])
    return s2

def Xgradient(X,Qs,Ds):
    """\
    Returns X gradient of multiview-MDSq stress.

    --- arguments ---
    X : data positions (n x p)
    Qs : list of orthogonal matrices (k x p x rank)
    Ds : list of target distance matrices (k x n x n)
    """
    dX = np.zeros(X.shape)
    for i in range(len(Qs)):
        dX += mdsq.Xgradient(X,Qs[i],Ds[i])
    return dX

def Xdescent(X0,Qs,Ds,feedback=False,**kwargs):
    """\
    X solution to multiview-MDSq problem using gradient descent.

    --- arguments ---
    X0 : initial positions, organized by row (n x p)
    Qs = list of orthogonal matrices
    Ds : list of target distance matrices (n x n)
    """
    if feedback is True:
        print("\nmmdsq.Xdescent():")
        
    df = lambda x: Xgradient(x,Qs,Ds)
    f = lambda x: stress(x,Qs,Ds)
    results = gd.gradient_descent(X0,df,f=f,feedback=feedback,**kwargs)
        
    return results

def Qdescent(X,Q0s,Ds,feedback=False,**kwargs):
    """\
    Solution to MDS problem using combined distance matrix. The list of distance
    matrices Ds is combined into one, and the node positions in R3 are computed
    using regular MDS, starting with initial configuration X0.

    --- arguments ---
    X = positions (n x p)
    Q0s = list of orthogonal matrices
    Ds = list of distance matrices (k x n x n)

    feedback = set to True to print computation feedback
    plot = set to True to print cost history
    """
    if feedback is True:
        print("\nmmdsq.Qdescent():")

    fs = lambda qs: stress(X,qs,Ds)

    def df_factory(i):
        def df(qs):
            return mdsq.Qgradient(X,qs[i],Ds[i])
        return df
    dfs = []
    for i in range(len(Q0s)):
        dfs.append(df_factory(i))

    p = lambda q: mdsq.nearest_orthogonal(q)

    results = gd.coordinate_gradient_descent(Q0s,dfs,fs=fs,projection=p,
                                             feedback=feedback,**kwargs)
    return results

def XQdescent(X0,Q0s,Ds,loops=1,rate=(0.01,0.01),feedback=False,**kwargs):
    if feedback is True:
        print("\nmmdsq.XQdescent():")

    fs = lambda xs: stress(xs[0],xs[1::],Ds)

    def df_factory(i):
        def df(xs):
            return mdsq.Qgradient(xs[0],xs[1+i],Ds[i])
        return df
    dfs = [lambda xs: Xgradient(xs[0],xs[1::],Ds)]
    for i in range(len(Q0s)):
        dfs.append(df_factory(i))

    p = [None]+[lambda q: mdsq.nearest_orthogonal(q)]*len(Q0s)

    if isinstance(rate,tuple):
        kwargs['rate'] = [rates[0]]+len(Q0s)*[rates[1]]
    else:
        assert rate > 0
        kwargs['rate'] = rate
    results = gd.coordinate_gradient_descent([X0]+Q0s,dfs,fs=fs,projection=p,
                                             loops=loops,feedback=feedback,
                                             **kwargs)
    return results

##### multiview MDS initializers #####

def combine_dmatrices(Ds):
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

def initiateX(Ds,X0=None,feedback=False,**kwargs):
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
        print("\nBegin mmdsq.initializeX():")

    n = len(Ds[0])
    #params = {
    #    'rate' : None,
    #    'max_iters': 1000
    #}
    #params.update(kwargs)
    #if params['rate'] is None:
    #    params['rate'] = set_init_rates(n)[0]
        
    D = combine_dmatrices(Ds)
    if X0 is None:
        X0 = random_ssphere(n)
        
    X = mds.gradient_descent(D,X0,feedback=feedback,**params)

    if feedback is True:
        print("\nEnd mmdsq.initializeX()")
        
    return X

def initializeQ(Ds,X,feedback=False,**kwargs):
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
    #params = {
    #    'rate' : None,
    #    'max_iters': 1000
    #}
    #params.update(kwargs)
    #if params['rate'] is None:
    #    params['rate'] = set_init_rates(n)[1]

    Q0s = standard_orthogonal(3,2)
    Qs = []
    for i in range(3):
        Q = mdsq.Qdescent(X,Q0s[i],Ds[i],feedback=feedback,**params)
        Qs += [Q]

    if feedback is True:
        stress0 = stress(X,Q0s,Ds)
        print(f"\n initial stress = {stress0:.2e}")
        stress1 = stress(X,Qs,Ds)
        print(f" final stress = {stress1:.2e}")
        print("\nEnd find_Q0().")
        
    return Qs

##### multiview MDS main algorithm #####

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

##### Tests #####

def example_Xdescent():
    """\
    mMDS X descend example, for exact points in cube and standard projections
    """
    print('\n##### TEST #####')
    print('mmdsq.example_Xdescent():')
    print('Recover points in cube from distance matrix of projected data for'\
          'the standard rank-2 projection matrices.')
    n=10
    X = np.random.rand(n,3)
    Qs = standard_orthogonal(3,2)
    Ds = dmatrices(X,Qs)

    X0 = np.random.rand(n,3)
    D0s = dmatrices(X0,Qs)
    stress0 = stress(X0,Qs,Ds)
    
    results = Xdescent(X0,Qs,Ds,rate=0.01,trajectory=True,
                       step_history=True,cost_history=True,
                       plot_history=True,feedback=True)
    Xf = results['output']
    stressf = stress(Xf,Qs,Ds)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xt = results['trajectory']
    for i in range(len(Xt)):
        plt.plot(Xt[i][:,0],Xt[i][:,1],Xt[i][:,2],'.',color='orange')
    ax.plot(X0[:,0],X0[:,1],X0[:,2],'o',color='y',label='initial')
    ax.plot(X[:,0],X[:,1],X[:,2],'*',color='g',label='original')
    ax.plot(Xf[:,0],Xf[:,1],X[:,2],'o',color='r',label='final')
    plt.legend()
    plt.title(f'MDS solution, initial stress = {stress0:.2e}, final stress = {stressf:.2e}')
    plt.show()

def example_Qdescent(trajectory=False):
    """\
    Multiview MDSq Q descend example, for exact points in cube and standard
    projections
    """
    print('\n##### TEST #####')
    print('mmdsq.example_Qdescent():')
    print('Recover rank-2 projections from points in cube and distance matrix '\
          'of projected data')
    n=10
    X = np.random.rand(n,3)
    Qs = standard_orthogonal(3,2)
    Ds = dmatrices(X,Qs)
    Q0s = random_orthogonal(3,2,number=3)
    stress0 = stress(X,Q0s,Ds)
    
    results = Qdescent(X,Q0s.copy(),Ds,loops=1,rate=0.01,
                       trajectory=trajectory,
                       cost_history=True,feedback=True)
    Qfs = results['output']
    stressf = stress(X,Qfs,Ds)
    print(f'  initial stress: {stress0}')
    print(f'  final stress: {stressf}')
    print(f'  initial grassmanian distances: {grassmanian_distance(Q0s,Qs)}')
    print(f'  final grassmanian distances: {grassmanian_distance(Qfs,Qs)}')

def example_XQdescent():
    print('\n##### TEST #####')
    print('mmdsq.example_XQdescent():')
    print('Recover both points in cube and rank-2 orthogonal projections from '\
          'distance matrix of projected data for.')
    n=10
    X = np.random.rand(n,3)
    Qs = standard_orthogonal(3,2)
    Ds = dmatrices(X,Qs)

    X0 = np.random.rand(n,3)
    Q0s = random_orthogonal(3,2,number=3)
    stress0 = stress(X0,Q0s,Ds)
    
    results = XQdescent(X0,Q0s,Ds,rate=0.01,trajectory=True,
                        feedback=True,loops=10,max_iters=100)
    xs = results['output']
    Xf = xs[0]
    Qfs = xs[1::]
    stressf = stress(Xf,Qfs,Ds)

    print('Qdescent_example0():')
    print(f'  initial stress: {stress0}')
    print(f'  final stress: {stressf}')
