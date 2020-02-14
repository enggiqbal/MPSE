import numbers, copy
import numpy as np
import networkx as nx

### Functions to set up a dissimilarity graph ###

# A dissimilarity graph is a dictionary containing the following attributes:
# 'edges' : a list or array containing the edges of the graph (each edge is a
# tuples of nodes)
# 'distances' : a list or array containing the edge distances of the given edges
# (it must have the same length as the list of edges)
# Other possible attributes are:
# 'weights' : a list or array containing the edge weights of the given edges (it# must hqve the same length as the list of edges)

def check(D, make_distances_positive=False):
    """\
    Takes dissimilarity graph or matrix and returns dissimilarity graph. If D is
    a dictionary, it checks that all attributes are included.
    """
    assert 'edges' in D
    assert 'distances' in D
    assert len(D['edges'])==len(D['distances'])

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

    weights : None or number or function or array_like
    If weights is None, w_ij = 1
    If weights is a number, w_ij = 1/Dij**int
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
        w = np.empty(NN)
        for i in range(NN):
            w[i] = weights(d[i])
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

def set_weights(D,function=None,scaling=0):
    """\
    Sets weights of dissimilarity graph as specified by given function or
    scalling number.

    D : dictionary
    Dissimilarity graph. Contains list of edges and distances.

    function : None or callable
    If a callable is given, assigns weight w=function(d) to an edge with 
    distance d.

    scaling : number
    If function is None, then assigns weights w=1/d**scaling to an edge with
    distance d.
    """
    distances = D['distances']; NN = len(distances)
    weights = np.ones(NN)
    if function is not None:
        assert callable(function)
        for nn in range(NN):
            weights[nn] = function(distances[nn])
    elif scaling != 0:
        assert isinstance(scaling,numbers.Number)
        for nn in range(NN):
            weights[nn] = distances[nn]**(-scaling)
    D['weights'] = weights

def remove_edges(D,number=None,proportion=0.2):
    """\
    Reduces number of edges in graph by eliminating far away neighbors.
    """
    d = D['distances']
    if number is not None:
        assert number < len(d)
    else:
        number = int(len(d)*proportion)
    ind = np.argpartition(d,number)[0:number]
    
    D = copy.deepcopy(D)
    D['edges'] = D['edges'][ind]
    D['distances'] = D['distances'][ind]
    D['weights'] = D['weights'][ind]

    return D

def sim2dict(S,mapping='reciprocal',connect_paths=None,connect_components=None):
    return

### GENERATORS ###

def generate_physical(N,dim=3):
    """\
    Generates a dissimilarity graph from the distances of coordinates.
    """
    X = misc.disk(N,dim=dim)
    D = from_coordinates(X)
    return D

def generate_binomial(N,p=0.1,distances=None):
    """\
    Generates a binomial graph (or Erdos-Renyi graph).

    Parameters:

    N : int
    Number of nodes.

    p : float, 0<p<=1
    Probability of edge creation

    distances : None or 'random'
    If None, distances are all one. If random, distances are distributed 
    uniformly at random between 0 and 1.
    """
    assert isinstance(p,float); assert 0<p<=1
    edges = []
    for i in range(N):
        for j in range(i+1,N):
            if np.random.rand() <= p:
                edges.append((i,j))
    if distances is None:
        distances = np.ones(len(edges))
    elif distances == 'random':
        distances = np.random.rand(len(edges))
    D = {
        'nodes' : range(N),
        'edges' : edges,
        'distances' : distances
        }
    return D

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

class DG(object):
    """\
    Class of dissimilarity graphs for a set of objects
    """

    def __init__(self,N,node_labels=None,dissimilarities=None):
        self.N = N
        
        if node_labels is None:
            node_labels = range(N)
        self.set_node_labels(node_labels)

        if dissimilarities is not None:
            K = len(dissimilarities)
            self.D = dissimilarities
            self.K = K

    def set_node_labels(self,node_labels):
        assert len(node_labels) == self.N
        self.node_labels = node_labels

    def from_perspectives(self,X,persp,**kwargs):
        Y = persp.compute_Y(X)
        D = []
        for y in Y:
            D.append(from_coordinates(y,**kwargs))
        self.D = D
        self.K = len(D)

    def generate_binomial(self,K=1,p=0.1,distance=None,**kwargs):
        D = []
        for k in range(K):
            D.append(generate_binomial(self.N,p=p,distances=None))
        self.D = D
        self.K = K
        
    def average_distances(self):
        return

### TESTS ###

def test_florence_distances():
    S = np.load('../multigraphs/florence/similarity_matrices.npy')
    #print(S)
    D = sim2dist(S[0])
    print(D)

if __name__ == '__main__':
    test_florence_distances()
