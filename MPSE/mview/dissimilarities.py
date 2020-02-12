import numbers
import numpy as np
import networkx as nx

### Functions to set up dissimilarity dictionary ###

def check(D, make_distances_positive=False):
    """\
    Takes dissimilarity graph or matrix and returns dissimilarity graph. If D is
    a dictionary, it checks that all attributes are included.
    """
    assert 'nodes' in D
    assert 'edges' in D
    assert 'distances' in D
    assert 'weights' in D
    assert len(D['edges'])==len(D['distances'])
    assert len(D['edges'])==len(D['weights'])


def from_coordinates(X,norm=2,edges=None,weights=None,colors=None):
    """\
    Returns dictionary with dissimilarity measures from coordinates.

    Parameters :

    X : (N,dimX) array_like
    Array containing coordinates of N points of dimension dimX.

    norm : number (>=1) or function
    If isinstance(norm,Number), this is the Minkowski p-norm with p=norm.
    If callable(norm), then use this as a norm.

    edges : None or float or array_like
    If edges is None: all edges are included.
    If isinstance(edges,Number): only edges with distance<edges are included.
    if isinstance(edges,array_like): this is the list of edges.

    weights : None or 'relative' or function or array_like
    If weights == None, w_ij = 1
    If weights == 'relative', w_ij = 1/D_ij^2
    If callable(weights), w_ij = weights(D_ij)
    If array_like, w_ij = weights[i,j]
    """
    N = len(X)
    if isinstance(norm,numbers.Number):
        p = norm
        assert p >= 1
        norm = lambda x: np.sum(x**p)**(1.0/p)
    else:
        assert callable(norm)

    if edges is None or isinstance(edges,numbers.Number):
        NN = int(N*(N-1)/2)
        e = np.empty((NN,2),dtype=int)
        d = np.empty(NN)
        if edges is None:
            it = 0
            for i in range(N):
                for j in range(i+1,N):
                    e[it] = [i,j]
                    d[it] = norm(X[i]-X[j])
                    it += 1
        else:
            it = 0
            for i in range(N):
                for j in range(N):
                    Dij = norm(X[i]-X[j])
                    if Dij <= edges:
                        e[it] = [i,j]
                        d[it] = norm(X[i]-X[j])
                        it += 1
            NN = it
            e = e[0:NN]
            d = d[0:NN]
    else:
        NN = len(edges)
        e = np.array(e,dtype=int)
        d = np.empty(NN)
        for i in range(NN):
            d[i] = norm(X[e[i,0]]-X[e[i,1]])

    if weights is None:
        w = np.ones(NN)
    elif weights == 'relative':
        w = d**(-2)
    elif callable(weights):
        for i in range(NN):
            w[i] = weights[d[i]]
    else:
        w = weights

    if colors is not None:
        if isinstance(colors,int):
            colors = misc.labels(Y,axis=colors)
    DD = {
        'nodes' : range(N),
        'edges' : e,
        'distances' : d,
        'weights' : w,
        'colors' : colors
        }
    return DD

def from_matrix(D,weights=None):
    """\
    Returns diccionary with dissimilarity relations from dissimilarity matrix.
    
    Parameters:

    D : (N,N) array_like
    Dissimilarity matrx.

    weights : None or 'relative' or function or array_like
    If weights == None, w_ij = 1
    If weights == 'relative', w_ij = 1/D_ij^2
    If callable(weights), w_ij = weights(D_ij)
    If array_like, w_ij = weights[i,j]
    """
    N = len(D); NN = N*(N-1)/2
    
    e = np.empty((NN,2),dtype=int)
    d = np.empty(NN)
    it = 0
    for i in range(N):
        for j in range(i+1,N):
            e[it] = [i,j]
            d[it] = D[i,j]
            it += 1

    w = np.empty(NN)
    if weights is None:
        w = np.ones(NN)
    elif weights == 'relative':
        it = 0
        for i in range(N):
            for j in range(i+1,N):
                w[it] = 1/D[i,j]**2
                it += 1
    elif callable(weights):
        it = 0
        for i in range(N):
            for j in range(i+1,N):
                w[it] = weights(D[i,j])
                it += 1
    else:
        it = 0
        for i in range(N):
            for j in range(i+1,N):
                w[it] = weights[i,j]
                it += 1

    DD = {
        'edges' : e,
        'dissimilarities' : d,
        'weights' : w
        }
    return DD

def remove_edges(D,number=None,proportion=0.2):
    """\
    Reduces number of edges in graph by eliminating far away neighbors.
    """
    d = D['distances']
    if number is not None:
        assert number < len(d)
    else:
        number = int(len(d)*proportion)
    ind = np.argpartition(d,number)
    
    D = copy.deepcopy(D)
    D['edges'] = D['edges'][ind]
    D['distances'] = D['distances'][ind]
    D['weights'] = D['weights'][ind]

    return D

def sim2dict(S,mapping='reciprocal',connect_paths=None,connect_components=None):
    return

### GENERATORS ###

### OLDER ###

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
