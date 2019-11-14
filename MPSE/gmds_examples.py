import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gmds import mds, special
import gmds_call

def onetwothree(n=30):
    """\
    Multiview MDS optimization for both X and Q

    n is number of points sampled from data set 123
    """
    factor = 30 #rough scaling of data

    path = 'dataset_3D/123_dataset_new/150/'
    D1 = np.genfromtxt(path+'data_mat_1_150.csv', delimiter=',')
    D2 = np.genfromtxt(path+'data_mat_2_150.csv', delimiter=',')
    D3 = np.genfromtxt(path+'data_mat_3_150.csv', delimiter=',')

    sub = np.random.choice(150,n,replace=False) #subsample of data
    D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub];
    Ds = [D1,D2,D3] #list containing distance matrices


    X,proj,cost,costhistory,Qs,X0,proj0,Q0s = gmds_call.main(D1,D2,D3,
                                                             feedback=True)

    W0 = special.normal_vectors(Q0s)
    W = special.normal_vectors(Qs)

    fig = plt.figure()
    plt.plot(costhistory)
    plt.title('stress per loop')
    plt.show(block=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X0[:,0],X0[:,1],X0[:,2],c='b',marker='o')
    for k in range(3):
        w = W0[k]*20
        ax.plot([0,w[0]],[0,w[1]],[0,w[2]])
    plt.title('initial')
    plt.show(block=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c='b',marker='o')
    for k in range(3):
        w = W[k]*20
        ax.plot([0,w[0]],[0,w[1]],[0,w[2]])
    plt.title('final')
    plt.show()

onetwothree(n=60)
