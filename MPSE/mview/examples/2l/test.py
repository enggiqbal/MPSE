import sys, os
sys.path.insert(0,os.getcwd()+'/../../../')
print(os.getcwd()+'/../')
import numpy as np
import matplotlib.pyplot as plt
import math, random
import mview

def test0(N=100):
    X = np.zeros((N,1))
    X[:,0] = np.random.rand(N)
    D = mview.distances.compute(X)
    color = mview.misc.labels(X)
    mv = mview.mds.MDS(D,dim=2,verbose=1,title='line example')
    mv.initialize()
    mv.approximate(verbose=1,batch_size=4)
    mv.approximate(verbose=1,batch_size=10)
    mv.optimize()
    mv.figure(labels=color)
    plt.show()

def set1(N=100,separation=1.0,true_distance=False):
    """\
    The data consits of samples of two parallel line segments.
    """
    X = np.zeros((N,2))
    X[:,0] = np.random.rand(N)
    X[int(N/2)::,1] = separation

    if true_distance is True:
        D = mview.distances.compute(X)
    else:
        D = np.zeros((N,N))
        D[0:int(N/2),0:int(N/2)] = mview.distances.compute(X[0:int(N/2)])
        D[int(N/2)::,int(N/2)::] = mview.distances.compute(X[int(N/2)::])
        D[0:int(N/2),int(N/2)::] = separation
        D[int(N/2)::,0:int(N/2)] = separation

    temp = sorted(X[:,0])     
    color = [temp.index(i) for i in X[:,0]] 
    return D, color

def test1(N=100,separation=1.0,true_distance=False):
    D, color = set1(N,separation,true_distance)
    
    persp = mview.perspective.Persp()
    persp.fix_Q(number=2,special='standard')
    mv = mview.mds.MDS(D,dim=2,verbose=1,title='2 lines example')
    mv.initialize()
    mv.approximate(verbose=1,batch_size=4)
    mv.approximate(verbose=1,batch_size=10)
    mv.optimize()
    mv.figure(labels=color)
    plt.show()

test0(N=100)
test1(N=100,separation=3,true_distance=False)
