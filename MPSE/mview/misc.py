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

### linear separation ###

## X -> dataset y-> labels either 0 or 1.
def computErrorLinearSeparatorSVM(X, y, plot=False):

    from sklearn import svm
    from sklearn.datasets import make_blobs
    
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    predicted = clf.predict(X)
    
    if plot is True:
        ## uncomment this to show the plots
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        # plot the decision function
        ax = plt.gca()
        print(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ##create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        ## plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        plt.show()

    return min(np.linalg.norm([1]*len(predicted) - predicted - y),
               np.linalg.norm(predicted - y))


# we create 40 separable points
if __name__=='__main__':
    import samples
    data = samples.sload('clusters2', n_samples=40)
    X = data['D']; yLabels = data['colors']
    errorVal = computErrorLinearSeparatorSVM(X, yLabels, plot=True)
    print(errorVal)
