##### Miscelaneous functions #####

import matplotlib.pyplot as plt
import numpy as np

### Functions to produce random initial embeddings ###

def box(number,dim=2,center=0,radius=1.0):
    X = 2*(np.random.rand(number,dim)-0.5)*radius+center
    return X

def disk(number,dim=2,center=0,radius=1.0):
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
