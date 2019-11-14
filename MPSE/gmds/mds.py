import numpy as np
import matplotlib.pyplot as plt

import gd

### Distance matrix ###

def dmatrix(X):
    """\
    Return distance matrix of nodes with positions given by array X.

    --- arguments ---
    X = numpy array containing node positions, organized by row.
    """
    from scipy.spatial import distance_matrix
    return distance_matrix(X,X)

### MDS optimization ###

def stress(X,D):
    """\
    Returns MDS stress between dmatrix(X) and D.

    --- arguments ---
    X : node positions, organized by row (n x p)
    D : target distance matrix (n x n)
    """
    n,p = X.shape
    d = dmatrix(X)

    stress = 0
    for i in range(n):
        for j in range(i+1,n):
            stress += (D[i,j]-d[i,j])**2

    return stress

def gradient(X,D):
    """\
    Returns gradient matrix of MDS stress at given node positions

    --- arguments ---
    X : positions, organized by row (n x p)
    D : target distance matrix (n x n)
    """
    n,p = X.shape
    d = dmatrix(X)
    
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            c = 2*(d[i,j]-D[i,j])/d[i,j]
            A[i,i] += c
            A[j,j] += c
            A[i,j] += -c
            A[j,i] += -c
    R = np.matmul(A,X)
    
    return R

def gradient_descent(D,X0,feedback=False,plot=False,**kwargs):
    """\
    Solution to MDS problem using gradient descent

    --- arguments ---
    D : target distance matrix (n x n)
    X0 : initial positions, organized by row (n x p)
    feedback = prints feedback if set to True
 
    --- kwargs ---
    rate, max_iters, min_step, max_step, trajectory, costs
    """
    if feedback is True:
        print("\nmds.gradient_descent():")
    if plot is True:
        kwargs['step_history'] = True
        kwargs['cost_history'] = True
        
    df = lambda x: gradient(x,D)
    f = lambda x: stress(x,D)
            
    results = gd.gradient_descent(X0,df,f=f,feedback=feedback,plot_history=plot,
                                  **kwargs)
        
    return results

##### Tests #####

def example_simple(rate=0.01):
    """\
    MDS algorithm for exact points in plane
    """
    print('\n##### TEST #####')
    print('mds.example_simple():')
    print('Points in plane are recovered from distance matrix')
    n=10
    X = np.random.rand(n,2)
    D = dmatrix(X)

    X0 = np.random.rand(n,2)
    stress0 = stress(X0,D)
    
    results = gradient_descent(D,X0,rate=rate,trajectory=True,
                              feedback=True,plot=True)
    Xf = results['output']
    stressf = stress(Xf,D)
    
    plt.figure()
    Xt = results['trajectory']
    for i in range(len(Xt)):
        plt.plot(Xt[i][:,0],Xt[i][:,1],'.',color='orange')
    plt.plot(X[:,0],X[:,1],'*',color='g',label='original')
    plt.plot(X0[:,0],X0[:,1],'o',color='y',label='initial')
    plt.plot(Xf[:,0],Xf[:,1],'o',color='r',label='final')
    plt.legend()
    plt.title(f'MDS solution, initial stress = {stress0:.2e}, '\
              f'final stress = {stressf:.2e}')
    plt.show()

def example_rate(n=10,p=3,min_rate=-8,max_rate=2):
    """\
    Run MDS for different choices of learning rate.
    X and X0 are uniformly random.
    """
    print('\n##### TEST #####')
    print('mds.example_rate():')
    print('Points in cube are recovered from distance matrix, for different'\
          'learning rates.')
    rates = 2.0**np.arange(min_rate,max_rate)
    
    X = np.random.rand(n,p)
    D = dmatrix(X)
    X0 = np.random.rand(n,p)

    plt.figure()
    for i in range(len(rates)):
        results = gradient_descent(D,X0,rate=rates[i],feedback=True,
                                   cost_history=True)
        fx = results['cost_history']
        plt.plot(fx,label=f'2^{min_rate+i}')

    plt.title(f'MDS stress for different learning rates, n = {n}, p = {p}')
    plt.xlabel('iteration number')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
def example_initial(n=10,p=3,rate=0.05,runs=10):
    """\
    Convergence of MDS algorithm for uniformly sampled points for different
    initial conditions.

    n : number of points
    p : dimension
    runs : number of runs
    """
    print()
    print('\n### Begin MDS_convergence_initial ###')
    X = np.random.rand(n,p)
    D = dmatrix(X)

    plt.figure()
    for i in range(runs):
        print(f'run {i+1} out of {runs}')
        X0 = np.random.rand(n,p)
        results = gradient_descent(D,X0,rate=rate,cost_history=True)
        fx = results['cost_history'] 
        plt.plot(fx)
    
    plt.title(f'MDS stress for various initial conditions, n = {n}, p = {p}, rate = {rate:.2f}.')
    plt.xlabel('iteration number')
    plt.ylabel('stress')
    plt.yscale('log')
    plt.show()
    print('### End MDS_convergence_initial ###')

    
############################ OLD

### multiview MDSq X and Q optimization ###

def mMDSq_stress(X,QQ,DD):
    """\
    Multiview-MDS stress for the Q-formulation

    --- variables ---
    X : data positions (n x p)
    QQ : list of orthogonal matrices (k x p x rank)
    DD : list of target distance matrices (k x n x n)
    """
    K = len(QQ)
    stress = 0
    for k in range(K):
        stress += MDS_stress(X @ (QQ[k] @ QQ[k].T), DD[k])
    return stress

def mMDSq_Xgradient(X,QQ,DD):
    """\
    Returns X gradient of multiview-MDS stress in Q-formulation

    --- variables ---
    X : data positions (n x p)
    QQ : list of orthogonal matrices (k x p x rank)
    DD : list of target distance matrices (k x n x n)
    """
    K = len(QQ)
    dX = np.zeros(X.shape)
    for k in range(K):
        dX += MDSp_Xgradient(X,QQ[k] @ QQ[k].T,DD[k])
    return dX

def mMDSq_Xdescent(X0,Q_list,D_list,feedback=False,trajectory=False,**kwargs):
    """\
    Gradient descent for multiview-MDS with fixed orthogonal projections

    --- arguments ---
    X0 : initial data positions (n x p)
    Q_list : list of transformation matrices (k x p x p)
    D_list : list of target distance matrices (k x n x n)

    feedback : return feedback if True
    trajectory : return trajectory if True

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion
    """
    if feedback is True:
        print("\nBegin mMDSq_Xdescent():")
        print(f"initial stress = {mMDSq_stress(X0,Q_list,D_list):.2e}")

    K = len(D_list)
    P_list = []
    for k in range(K):
        Q = Q_list[k]
        P_list += [Q @ Q.T]
        
    X = mMDSp_Xdescent(X0,P_list,D_list,feedback=False,trajectory=False,
                       **kwargs)

    if feedback is True:
        print(f"final stress = {mMDSq_stress(X,Q_list,D_list):.2e}")
        print("End mMDSq_Xdescent():")
        
    return X

def mMDSq_XQdescent(X0,QQ0,DD,feedback=False,**kwargs):
    """\
    Multiview-MDS optimization for X and Q using coordinate projected gradient
    descent 

    --- arguments ---
    X0 : initial data positions (n x p)
    PP0 : list of initial maps (k x p x p)
    DD : list of target distance matrices (k x n x n)

    feedback : return feedback if True

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion
    """
    if feedback is True:
        print("\nBeginning multiview MDS X and Q descent")
        print("initial stress = ", mMDSq_stress(X0,QQ0,DD))

    XX0 = [X0]+QQ0 #list containing initial data for all coordinates
    dff = []
    dX = lambda XX: mMDSq_Xgradient(XX[0],XX[1::],DD)
    dff += [dX]
    K = len(QQ0);
    for k in range(K):
        dPk = lambda XX: MDSq_Qgradient(XX[0],XX[1+k],DD[k])
        dff += [dPk]
    identity = lambda X: X
    pp = [identity] + K*[projection_to_orthogonal]
    
    XX = gd.coord_projected_gradient_descent(dff,pp,XX0,**kwargs)
    X = XX[0]
    QQ = XX[1::]
    
    if feedback is True:
        print("final stress = ",mMDSq_stress(X,QQ,DD))
        
    return X,QQ

