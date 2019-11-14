import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

import gd, mds, mdsq, mmdsq

X = np.load('example123/X.npy')
Y1 = np.load('example123/1.npy')
Y2 = np.load('example123/2.npy')
Y3 = np.load('example123/3.npy')

D1 = mds.dmatrix(Y1)
D2 = mds.dmatrix(Y2)
D3 = mds.dmatrix(Y3)
Ds = [D1,D2,D3]

Qs = mmdsq.cylinder(3)
#Ds = mmdsq.dmatrices(X,Qs)####
n = 100

#Y1_test = mdsq.projected_positions(X,Qs[0])
#plt.figure()
#plt.plot(Y1_test[:,0],Y1_test[:,2],'o')
#plt.show()

### X descent only ###

def findX():
    X0 = np.random.rand(n,3)-0.5
    stress0 = mmdsq.stress(X0,Qs,Ds)
    results = mmdsq.Xdescent(X0,Qs,Ds,rate=0.001,trajectory=True,
                             step_history=True,cost_history=True,
                             plot_history=True,feedback=True)
    Xf = results['output']
    stressf = mmdsq.stress(Xf,Qs,Ds)
    print('initial stress=',stress0)
    print('final stress=',stressf)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X0[:,0],X0[:,1],X0[:,2],'o',color='y',label='initial')
    ax.plot(X[:,0],X[:,1],X[:,2],'*',color='g',label='original')
    ax.plot(Xf[:,0],Xf[:,1],X[:,2],'o',color='r',label='final')
    plt.legend()
    plt.title(f'MDS solution, initial stress = {stress0:.2e}, '\
              +f'final stress = {stressf:.2e}')
    plt.show()

def findQ():
    Q0s = mmdsq.random_orthogonal(3,2,3)
    stress0 = mmdsq.stress(X,Q0s,Ds)
    results = mmdsq.Qdescent(X,Q0s,Ds,rate=0.001,feedback=True)
    Qfs = results['output']
    stressf = mmdsq.stress(X,Qfs,Ds)
    print('initial stress=',stress0)
    print('final stress=',stressf)

    Yfs = mmdsq.projected_positions(X,Qfs)
    fig, axs = plt.subplots(1,3,sharex=True)
    plt.tight_layout()
    for i in range(3):
        axs[i].scatter(Yfs[i][:,0],Yfs[i][:,2])
        axs[i].set_aspect(1.0)
        axs[i].set_title(f'Projection {i}')
    plt.suptitle('Projected Data')
    plt.show()

def findXQ():
    X0 = np.random.rand(n,3)-0.5
    Q0s = mmdsq.random_orthogonal(3,2,3)
    stress0 = mmdsq.stress(X0,Q0s,Ds)
    results = mmdsq.XQdescent(X0,Q0s,Ds,rate=0.001,feedback=True,loops=5)
    Xf = results['output'][0]
    Qfs = results['output'][1:]
    stressf = mmdsq.stress(X,Qfs,Ds)
    print('initial stress=',stress0)
    print('final stress=',stressf)

    Yfs = mmdsq.projected_positions(Xf,Qfs)
    fig, axs = plt.subplots(1,3,sharex=True)
    plt.tight_layout()
    for i in range(3):
        axs[i].scatter(Yfs[i][:,0],Yfs[i][:,2])
        axs[i].set_aspect(1.0)
        axs[i].set_title(f'Projection {i}')
    plt.suptitle('Projected Data')
    plt.show()

findXQ()
