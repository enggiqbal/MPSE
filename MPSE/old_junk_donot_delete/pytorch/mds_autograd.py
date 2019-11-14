import matplotlib.pyplot as plt
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

alpha=0.0001
steps=1200
eps=1e-9
dim=2

import sys
sys.path.append('../dataset')
import graph_similarity_matrix as gsm
import data as mdata



def data():
    M=np.array([[12,10,20,30,40,50], [16,40,250,60,70,80],[21,10,311,12,13,14],[15,415,21,42,53,64],[21,410,121,12,13,14],[1,155,21,42,533,64]], dtype=float)
#    M=np.random.rand(100,100)
    for i in range(0,M.shape[0]):
        for j in range(i, M.shape[1]):
            if i==j:
                M[i][j]=0.0
            else:
                M[i][j]=float( M[j][i])
    return M

def mds_sklearn(alpha,A,M,steps,dim):
    m=int(len(A)/dim)
    X=A.reshape(m,dim)
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=dim, max_iter=steps, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1, verbose=3)
    #init=Xzz
    pos = mds.fit(M).embedding_
    return pos.reshape(m*dim,1)




def mds_autograd(alpha,A,M,steps,dim):
    g=grad(costfunction)
    for i in range(0,steps):
        A=A- alpha * g(A)
        print("step: ", i, ", cost:", costfunction(A))
    return A


def costfunction(A):
    """
    cost function of MDS
    """
    global M
    global eps
    global dim
    #global dis
    #return np.sum(A)
    m=int(len(A)/dim)
    X=A.reshape(m,dim)
#    dis=cosine_similarity(X)
#    dis=X@X.T
    cost=0

    for i in range(0, m):
        for j in range(i+1, m):
            #print("Xi==",X[i])
            diff=np.square( X[j][0]-X[i][0]) + np.square( X[j][1]-X[i][1])
            if diff<eps:
                d=0
            else:
                d= np.sqrt(diff)
            #print("diss",d)
            if abs(d - M[i][j]) > eps:
                cost = cost+ np.square(d - M[i][j])
            #dis[j][i]=dis[i][j]
    #dis = euclidean_distances(X)
    #cost1=np.square(dis-M)
    #cost=np.sum(cost1)
    return cost


#dotpath='../dataset/game_of_thrones_consistent.dot'
#M,  G, nodes_index=gsm.get_similarity_matrix(dotpath)

M=mdata.get_matrix('../dataset/dist_2.csv')
M=M[0:29,0:29]
print(M.shape)
#X=np.zeros((6,2))
#X=np.random.rand(6,2)

#M=data()

A=np.random.rand(len(M)*dim,1)
#B=A.copy()
#print(A)
ZZ=mds_autograd(alpha,A,M,steps,dim)


Z=mds_sklearn(alpha,A,M,steps,dim)



fig = plt.figure()
ax = plt.axes()

pos1=ZZ.reshape(int(len(ZZ)/dim),dim)
X,Y=pos1.T[0], pos1.T[1]
ax.scatter(X, Y,  c='red', cmap='Greens');

pos2=Z.reshape(int(len(Z)/dim),dim)
X,Y=pos2.T[0], pos2.T[1]
ax.scatter(X, Y,  c='green', cmap='Greens');
for i in range(0,len(pos1)):
    line = mlines.Line2D([pos1[i][0],pos2[i][0]], [pos1[i][1],pos2[i][1]], color='red')
    ax.add_line(line)

plt.show()
