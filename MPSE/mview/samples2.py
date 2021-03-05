import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance

import misc, setup, multigraph, gd, projections, mds, tsne, plots

def disk(n_samples=1000):
    X = misc.disk(n_samples, dim=3)
    proj = projections.PROJ()
    Q = proj.generate(number=3, method='standard')
    Y = proj.project(Q,X)
    return Y, X, Q

def e123():
    X = np.genfromtxt('samples/123/123.csv',delimiter=',')
    X1 = np.genfromtxt('samples/123/1.csv',delimiter=',')
    X2 = np.genfromtxt('samples/123/2.csv',delimiter=',')
    X3 = np.genfromtxt('samples/123/3.csv',delimiter=',')
    proj = projections.PROJ()
    Q = proj.generate(number=3,method='cylinder')
    return [X1,X2,X3], X, Q
    
def cluster():
    import csv
    path = 'samples/cluster/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'dist_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        Y.append(array)
    labels = open(path+'labels.csv')
    labels = np.array(list(csv.reader(labels)),dtype=int).T
    return Y, labels

def florence():
    sys.path.insert(1,'samples/florence/')
    import setup_florence as setup
    return setup.setup2()

def credit():
    import csv
    path = 'samples/credit/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'discredit3_tsne_cluster_1000_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        array += np.random.randn(len(array),len(array))*1e-4
        Y.append(array)
    return Y
