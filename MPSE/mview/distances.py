import numpy as np
import networkx as nx

def coord2dist(X,p=2):
    """\
    Returns distance matrix for a set of points with coordinates X.

    --- Parameters ---

    X : (N,dimX) array_like
    Array containing coordinates of N points of dimension dimX.

    p : float, 1 <= p <= Inf
    Which Minkowski p-norm to use.
    
    --- Returns ---

    D : (N,N) ndarray
    Array containing pairwise distances between points in X.
    """
    from scipy.spatial import distance_matrix
    D = distance_matrix(X,X,p)
    return D

def sim2dist(S,shortest_path=True,connect_components=True,connect_factor=2.0):
    assert isinstance(S, np.ndarray); assert (S>=0).all()

    N = len(S)
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j != i:
                if S[i,j] == 0:
                    D[i,j] = np.nan
                else:
                    D[i,j] = 1.0/S[i,j]

    if shortest_path is True:
        G = nx.Graph()
        for i in range(N):
            for j in range(N):
                if np.isnan(D[i,j]) == False:
                    G.add_edge(i, j, weight=D[i,j])
        paths = dict(nx.shortest_path_length(G,weight='weight'))
        for i in range(N):
            for j in range(N):
                if j in paths[i].keys():
                    D[i,j] = paths[i][j]

    if connect_components is True:
        max_dist = np.nanmax(D)
        for i in range(N):
            for j in range(N):
                if np.isnan(D[i,j]):
                    D[i,j] = connect_factor*max_dist

    return D

def dmatrices(X,input_type='coordinates',**kwargs):
    assert input_type in ['coordinates','similarities']
    K = len(X); N = len(X[0])
    D = np.empty((K,N,N))
    if input_type is 'coordinates':
        for k in range(K):
            D[k] = coord2dist(X[k],**kwargs)
    elif input_type is 'similarities':
        for k in range(K):
            D[k] = sim2dist(X[k],**kwargs)
    return D

### Main function ###

input_types = ['coordinates','similarities']

algorithms = {
    'coordinates' : coord2dist,
    'similarities' : sim2dist
    }

def compute(X,input_type='coordinates',**kwargs):
    assert input_type in input_types
    algorithm = algorithms[input_type]
    if isinstance(X,np.ndarray) and len(X.shape)==2:
        D = algorithm(X,**kwargs)
    elif isinstance(X,np.ndarray) and len(X.shape)==3:
        K,N,_ = X.shape
        D = np.empty((K,N,N))
        for k in range(K):
            D[k] = algorithm(X[k],**kwargs)
    elif isinstance(X,list):
        D = []
        for x in X:
            D.append(algorithm(x,**kwargs))
    else:
        sys.exit('Incorrect form.')
    return D

def clean(D,epsilon=1e-3,verbose=0):
    if np.sum(D==0)>len(D):
        D += epsilon
        if verbose > 0:
            print(f'  distances.clean(): {epsilon:0.2e} added to D')

##### Functions to add noise to distance/dissimilarity matrices #####

def add_relative_noise(D,sigma):
    """\
    Returns a noisy copy of a distance/dissimilarity matrix D. The error is 
    relative, meaning that an entry D[i,j] of D is perturbed by gaussian noise 
    with standard deviation sigma*D[i,j]. The noisy matrix remains symmetric.
    """
    N = len(D); D_noisy = D.copy()
    for i in range(N):
        for j in range(i+1,N):
            noise_factor = 1+sigma*np.random.randn()
            D_noisy[i,j] *= noise_factor
            D_noisy[j,i] = D_noisy[j,i]
    return D_noisy

def add_noise(D,sigma,noise_type='relative'):
    if noise_type == 'relative':
        noise_function = add_relative_noise

    if isinstance(D,np.ndarray) and len(D.shape)==2:
        D_noisy = noise_function(D,sigma)
    elif isinstance(D,np.ndarray) and len(D.shape)==3:
        K,N,_ = D.shape
        D_noisy = np.empty((K,N,N))
        for k in range(K):
            D_noisy[k] = noise_function(D[k],sigma)
    elif isinstance(D,list):
        K = len(D)
        D_noisy = []
        for k in range(K):
            D_noisy.append(noise_function(D[k],sigma))
    else:
        sys.exit('Incorrect form.')
    return D_noisy

### TESTS ###

def test_florence_distances():
    S = np.load('../multigraphs/florence/similarity_matrices.npy')
    #print(S)
    D = sim2dist(S[0])
    print(D)

if __name__ == '__main__':
    test_florence_distances()
