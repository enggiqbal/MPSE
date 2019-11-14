import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
#import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad
import math
from sklearn.metrics import euclidean_distances, pairwise_distances
import autograd.numpy as np
from sklearn import manifold
import torch.nn.functional  as F
import matplotlib.lines as mlines



import sys
sys.path.append('../dataset')
import graph_similarity_matrix as gsm
import data as mdata


alpha=0.003
steps=1200
eps=1e-9
stopping_eps=0.1
dim=3
P1=np.array([[1,0,0], [0,0,0],[0,0,1]], dtype=float)
P2=np.array([[1,0,0], [0,1,0],[0,0,0]], dtype=float)
P3=np.array([[0,0,0], [0,1,0],[0,0,1]], dtype=float)

def data():
    D1=np.array([[12,10,20,30,40,50], [16,40,250,60,70,80],[21,10,311,12,13,14],[15,415,21,42,53,64],[21,410,121,12,13,14],[1,155,21,42,533,64]], dtype=float)
    D2=np.array([[12,120,20,30,40,50], [16,450,250,60,70,804],[21,10,611,12,13,14],[15,415,261,42,53,64],[21,410,1231,12,13,14],[1,155,214,42,533,634]], dtype=float)
    D3=np.array([[12,120,202,330,40,150], [16,450,2250,603,70,80],[21,10,611,12,13,14],[15,4515,21,42,53,64],[21,410,1231,12,13,14],[1,155,21,42,333,634]], dtype=float)
#    M=np.random.rand(100,100)
    for i in range(0,D1.shape[0]):
        for j in range(i, D1.shape[1]):
            D1[i][j]=0.0 if i==j else float( D1[j][i])
            D2[i][j]=0.0 if i==j else float( D2[j][i])
            D3[i][j]=0.0 if i==j else float( D3[j][i])
    return D1, D2, D3

def multiview_autograd(alpha,A,steps,dim, stopping_eps):
    g=grad(costfunction)
    oldcost= costfunction(A)
    for i in range(0,steps):
        A=A- alpha * g(A)
        newcost= costfunction(A)
        print("step: ", i, ", cost:", newcost)

        if oldcost-newcost < stopping_eps:
            print("early stopping at", i )
            return A
        oldcost=newcost

    return A

def costfunction(A):
    """
    stress function multiview
    """
    global D1, D2, D3
    global dim
    global P1, P2, P3
    m=int(len(A)/dim)
    X=A.reshape(m,dim)
    cost=0
    cost=projectionCost(X,P1, D1 )+ projectionCost( X,P2, D2 )# +projectionCost( X,P3, D3 )
    return cost

def projectionCost(X, P, W):
    cost=0
    m=len(X)
    for i in range(0, m):
        for j in range(i+1, m):
            vi=P @ X[i]
            vj=P @ X[j]
            diff=np.sum( np.square( vi-vj))
            d=0 if diff<eps else np.sqrt(diff)
            cost=cost+ np.square(d - W[i][j]) if  abs(d - W[i][j]) > eps else cost
        return cost





#dotpath='../dataset/game_of_thrones_consistent.dot'
#M,  G, nodes_index=gsm.get_similarity_matrix(dotpath)


#X=np.zeros((6,2))
#X=np.random.rand(6,2)


#D1, D2, D3=data()
D1=mdata.get_matrix('../dataset/dist_1.csv')
D2=mdata.get_matrix('../dataset/dist_2.csv')
#D3=mdata.get_matrix('../dataset/dist_2.csv')
D3=np.zeros((len(D1),len(D1)))

A=np.random.rand(len(D1)*dim,1)
#B=A.copy()
#print(A)
pos1=multiview_autograd(alpha,A,steps,dim, stopping_eps)
#pos2=mds_sklearn(alpha,A,steps,dim)


pos1=pos1.reshape(int(len(pos1)/dim),dim)
#pos2=pos2.reshape(int(len(ZZ)/dim),dim)

fig = plt.figure()
ax = plt.axes(projection='3d')

X,Y,Z=pos1.T[0], pos1.T[1], pos1.T[2]
ax.scatter3D(X, Y, Z, c='red', cmap='Greens');
#X,Y,Z=pos2.T[0], pos2.T[1], pos2.T[2]
#ax.scatter3D(X, Y, Z, c='red', cmap='Greens');

#for i in range(0,len(pos1)):
#    line = mlines.Line2D([pos1[i][0],pos2[i][0]], [pos1[i][1],pos2[i][1]], color='red')
#    ax.add_line(line)

plt.show()
