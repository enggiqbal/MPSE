import os, sys
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,directory)
import csv

import numpy as np

def disk(n_samples=1000):
    import misc, projections
    X = misc.disk(n_samples, dim=3)
    proj = projections.PROJ()
    Q = proj.generate(number=3, method='standard')
    Y = proj.project(Q,X)
    return Y, X, Q

def e123():
    X = np.genfromtxt(path+'123/123.csv',delimiter=',')
    X1 = np.genfromtxt(path+'123/1.csv',delimiter=',')
    X2 = np.genfromtxt(path+'123/2.csv',delimiter=',')
    X3 = np.genfromtxt(path+'123/3.csv',delimiter=',')
    proj = projections.PROJ()
    Q = proj.generate(number=3,method='cylinder')
    return [X1,X2,X3], X, Q
    
def cluster():
    import csv
    path = directory+'/cluster/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'dist_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        Y.append(array)
    labels = open(path+'labels.csv')
    labels = np.array(list(csv.reader(labels)),dtype=int).T
    return Y, labels

def florence():
    sys.path.insert(1,directory+'/florence')
    import setup_florence as setup
    return setup.setup2()

def credit():
    import csv
    path = directory+'/credit/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'discredit3_tsne_cluster_1000_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        array += np.random.randn(len(array),len(array))*1e-4
        Y.append(array)
    return Y

def phishing(groups=[0,1,2,3], n_samples=None):
    import phishing
    features = phishing.features
    labels = phishing.group_names

    if n_samples is None:
        n_samples = len(features[0])
    Y, perspective_labels = [], []
    for group in groups:
        assert group in [0,1,2,3]
        Y.append(features[group][0:n_samples])
        perspective_labels.append(labels[group])
        
    sample_colors = phishing.results[0:n_samples]
    return Y, sample_colors, perspective_labels

def mnist0():
    Y = []
    for ind in ['1','2']:
        filec = open(directory+'/MNIST/MNIST_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        Y.append(array)
    filec = open(directory+'/MNIST/MNIST_labels.csv')
    labels =  np.array(list(csv.reader(filec)),dtype='float')
    filec = open(directory+'/MNIST/MNIST_labels.csv')
    X = np.array(list(csv.reader(filec)),dtype='float')
    return Y, labels, X

def mnist(n_samples=1000):
    from keras.datasets import mnist
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    X = X_train[0:n_samples]
    labels = Y_train[0:n_samples]
    X = X.reshape(n_samples,28*28)
    X = np.array(X,dtype='float')/256
    return X, labels

def load(dataset, **kwargs):
    "returns dictionary with 
