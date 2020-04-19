##### Miscelaneous functions #####

import matplotlib.pyplot as plt
import numpy as np

### Functions to produce random initial embeddings ###

def box(number,dim=2,center=0,radius=1.0,**kwargs):
    X = 2*(np.random.rand(number,dim)-0.5)*radius+center
    return X

def disk(number,dim=2,center=0,radius=1.0,**kwargs):
    r = np.random.rand(number)
    X0 = np.random.randn(number,dim)
    X = (X0.T / np.linalg.norm(X0,axis=1)*r**(1.0/dim)).T*radius+center
    return X

initial_embedding_methods = {
    'box' : box,
    'disk' : disk
    }

def initial_embedding(number,method='disk',**kwargs):
    """\
    Produce initial embedding using methods above.
    """
    algorithm = initial_embedding_methods[method]
    X = algorithm(number,**kwargs)
    return X

### Function to produce labels ###

def labels(X,function=None,axis=0):
    if function is None:
        temp = sorted(X[:,axis])     
        labels = [temp.index(i) for i in X[:,axis]]
    return labels

### Function to label entries in upper triangular (without diagonal) ###

def list_to_triangular(N,index_list):
    edges = np.empty((len(index_list),2),dtype=int)
    i = N-2-np.floor(np.sqrt(-8*index_list+4*N*(N-1)-7)/2.0-0.5)
    j = index_list+i+1-N*(N-1)/2+(N-i)*((N-i)-1)/2
    edges[:,0] = i; edges[:,1] = j
    return edges

def random_triangular(N,number,replace=False):
    k = np.random.choice(round(N*(N-1)/2),number,replace=replace)
    edges = list_to_triangular(N,k)
    return edges
