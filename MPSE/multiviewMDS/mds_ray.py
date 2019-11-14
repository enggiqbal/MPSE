import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def gradient_descent(df,X0,rate=0.1,precision=0.001):
    """\
    Gradient descent algorithm
    """
    x = X0
    step_size = 1.0
    iters = 0
    max_iters = 1000
    while step_size > precision and iters < max_iters:
        x0 = x
        x = x - rate*df(x)
        step_size = np.linalg.norm(x-x0)
        iters += 1
    return x

def MDS_stress(X,D):
    n = len(X)
    DX = distance_matrix(X,X)

    stress = 0
    for i in range(n):
        for j in range(i+1,n):
            stress += (D[i,j]-DX[i,j])**2

    return stress

def MDS_gradient(X,D):
    """\
    Compute gradient matrix for MDS
    D is target distances
    X is current positions
    """
    n = len(X)
    d = distance_matrix(X,X)
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

def MDS_descent(D,X0,rate=0.01,precision=0.0001):
    """\
    Solution to MDS problem using gradient descent
    """
    df = lambda x: MDS_gradient(x,D)
    X = gradient_descent(df,X0,rate,precision)
    return X

def example1():
    """\
    Exact random points in plane
    """
    n = 10

    X = np.random.rand(n,2)
    D = distance_matrix(X,X)

    X0 = np.random.rand(n,2)
    X1 = MDS_descent(D,X0)

    stress = MDS_stress(X1,D)
    print("Stress = ",stress)

    plt.figure()
    plt.plot(X[:,0],X[:,1],'o',label='original')
    plt.plot(X0[:,0],X0[:,1],'o',label='initial')
    plt.plot(X1[:,0],X1[:,1],'o',label='final')
    plt.legend()
    plt.show()

def MDS2_stress(X,P_tensor,D_tensor):
    K = len(P_tensor)
    stress = 0
    for k in range(K):
        stress += MDS_stress(np.matmul(X,P_tensor[k]),D_tensor[k])
    return stress

def MDS2_gradient(X,P_tensor,D_tensor):
    (n,p) = X.shape
    K = len(P_tensor)

    dX = np.zeros((n,p))
    for k in range(K):
        P = P_tensor[k]
        D = D_tensor[k]
        dXk = MDS_gradient(np.matmul(X,P),D)
        dX += np.matmul(dXk,P.transpose())

    return dX

def MDS2_descent(P_tensor,D_tensor,X0,rate=0.01,precision=0.0001):
    """\
    Solution to MDS2 problem using gradient descent
    """
    df = lambda x: MDS2_gradient(x,P_tensor,D_tensor)
    X = gradient_descent(df,X0,rate,precision)
    return X

def MDS2_example1():
    n = 10

    X = np.random.rand(n,3)
    D = distance_matrix(X,X)

    P_tensor = np.empty((3,3,3))
    D_tensor = np.empty((3,n,n))

    for k in range(3):
        P = np.diag(np.ones(3))
        P[k,k] = 0
        P_tensor[k] = P
        XP = np.matmul(X,P)
        D_tensor[k] = distance_matrix(XP,XP)

    X0 = np.random.rand(n,3)
    X1 = MDS2_descent(P_tensor,D_tensor,X0)

    stress = MDS2_stress(X1,P_tensor,D_tensor)
    print("Stress = ",stress)

MDS2_example1()
example1()
