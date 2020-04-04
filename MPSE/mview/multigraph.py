import sys, numbers, copy, math
import numpy as np
import networkx as nx

import misc, projections
### Functions to set up distance graph and multigraphs ###

# A (distance) graph is a dictionary containing the following attributes:
# 'node_number' : number of nodes
# 'edge_number' : number of edges
# 'edges' : a list or array containing the edges of the graph
# 'distances' : a list or array containing the edge distances of the given edges
# (it must have the same length as the list of edges)
# Other possible attributes are:
# 'weights' : a list or array containing the edge weights of the given edges (it# must hqve the same length as the list of edges)

### Check and setup attribute or dissimilarity graph ###

def attribute_check(D):
    assert isinstance(D,dict)
    for key in ['nodes','edges','type','complete']:
        assert key in D
    if D['complete'] is True:
        assert 'dfunction' in D
    else:
        assert 'elist' in D
    if D['type'] == 'matrix':
        assert 'matrix' in D
    elif D['type'] == 'features':
        assert 'features' in D
    elif D['type'] == 'graph':
        'dlist' in D
    else:
        sys.abort('Attribute type incorrect')

def attribute_setup(D,**kwargs):
    """\
    Sets up attribute dictionary without having to call DISS explicitly.

    Parameters:

    D : array or dictionary
    """
    if isinstance(D,dict) and 'type' in D:
        D = D
    elif isinstance(D,np.ndarray):
        N = len(D)
        diss = DISS(N,**kwargs)
        if len(D.shape)==2 and D.shape[0]==D.shape[1]:
            diss.from_matrix(D,**kwargs)
        else:
            diss.from_features(D,**kwargs)
        D = diss.D[0]
    elif isinstance(D,dict):
        if 'nodes' in D:
            N = D['nodes']
        else:
            assert 'elist' in D
            N = np.max(D['elist'])
        diss = DISS(N,**kwargs)
        diss.from_graph(**D,**kwargs)
        D = diss.D[0]
    return D

def attribute_rms(D,estimate=True,**kwargs):
    if D['complete'] is True:
        rms = 0
        if estimate is True and D['nodes'] > 64:
            edges = misc.random_triangular(D['nodes'],int(64*63/2))
            for i1,i2 in edges:
                rms += D['dfunction'](i1,i2)**2
            rms = math.sqrt(rms/(64*63/2))
        else:
            for i in range(D['nodes']):
                for j in range(D['nodes']):
                    rms += D['dfunction'](i,j)**2
            rms = math.sqrt(rms/(D['nodes']*(D['nodes']-1)/2))
    else:
        if estimate is True and D['edges'] > 64*63/2:
            inds = np.random.choice(D['edges'],int(64*63/2))
            rms = np.linalg.norm(D['dlist'][inds])/math.sqrt(64*63/2)
        else:
            rms = np.linalg.norm(D['dlist'])/math.sqrt(D['edges'])
    return rms

def attribute_sample(D,edge_proportion=None,average_neighbors=None,
                     replace=True,**kwargs):
    if edge_proportion is None and average_neighbors is None:
        return D
    else:
        N = D['nodes']; NN0 = D['edges']
        if edge_proportion is not None:
            NN = round(edge_proportion*NN0)
        elif average_neighbors is not None:
            NN = min(round(average_neighbors*N/2),NN0)
        
        if D['complete'] is True:
            edges = misc.random_triangular(N,NN,replace=replace)
            dlist = np.empty(NN)
            for i in range(NN):
                edge = edges[i]
                dlist[i] = D['dfunction'](int(edge[0]),int(edge[1]))
        else:
            inds = np.random.choice(NN0,NN)
            edges = D['elist'][inds]
            dlist = D['dlist'][inds]

        Ds = {}
        Ds['nodes'] = N
        Ds['type'] = 'graph'
        Ds['complete'] = False
        Ds['edges'] = NN
        Ds['elist'] = edges
        Ds['dlist'] = dlist
        Ds['label'] = 'sample'
        Ds['weighted'] = False

        return Ds
    

def multigraph_check(DD):
    """\
    Checks/complete multigraph dictionary DD.
    """
    assert isinstance(DD,DISS)

def multigraph_setup(D0,**kwargs):
    """\
    Sets up distance multigraph from array or list or dictionary.

    If D0 is a an array or list of arrays, then it is assumed that it consists 
    of square matrices containing pairwise distances.
    
    If D0 is a dictionary, it is assumed that it already contains the distance
    graphs.
    """
    if isinstance(D0,DISS):
        DD = D0
    elif isinstance(D0,dict):
        assert 'nodes' in D0
        assert 'attributes' in D0
        DD = DISS(**D0)
        DD.from_projections(**D0,**kwargs)

    else:
        if isinstance(D0,np.ndarray):
            if len(D0.shape) <= 2:
                D0 = [D0]

        K = len(D0)
        D = []
        for k in range(K):
            D.append(attribute_setup(D0[k]))

        nodes = max([D[k]['nodes'] for k in range(K)])
        DD = DISS(nodes,**kwargs)
        DD.D = D
        DD.attributes = K

    return DD

### CLASS ###

class DISS(object):
    """\
    Class with methods to compute dissimilarity relations
    """
    def __init__(self, nodes, nlabel=None, ncolor=None,**kwargs):
        self.nodes = nodes
        self.nlabel = nlabel
        self.ncolor = ncolor
        self.attributes = 0
        self.D = []

    def return_attribute(self,attribute=0,**kwargs):
        """\
        Returns dictionary for specified attribute.
        """
        assert isinstance(attribute,int)
        assert attribute in range(self.attributes)
        D = self.D[attribute]
        #if 'ncolor' not in D or D['ncolor'] is None:
        #    D['ncolor'] = self.ncolor
        return D

    def from_matrix(self,matrix,label=None,**kwargs):
        """\
        Adds a perspective to self using a pairwise dissimilarity matrix.
        
        Parameters:
        matrix : (N by N) numpy array
        Dissimilarity matrix.
        """
        assert isinstance(matrix,np.ndarray)
        shape = matrix.shape; assert len(shape)==2;
        assert shape[0]==self.nodes; assert shape[1]==self.nodes

        D = {}
        D['nodes'] = self.nodes
        D['type'] = 'matrix'
        D['matrix'] = matrix
        D['complete'] = True
        D['edges'] = int(self.nodes*(self.nodes-1)/2)
        D['dfunction'] = lambda i,j : D['matrix'][i,j]                
        D['label'] = label
        D['ncolor'] = self.ncolor
        
        self.add_weights(D,**kwargs)
        self.D.append(D)
        self.attributes += 1

    def from_features(self,features,distance=None,label=None,**kwargs):
        """\
        Adds a perspective to self node features and a distance function.
        
        Parameters:
        
        features : (length N) list or array
        Contains node features.

        distance :None or callable
        If None, uses Euclidean distance (each feature must be np.ndarray).
        If callable, uses distance as pairwise distance function.
        """
        assert len(features) == self.nodes
        if distance is None:
            distance = lambda x,y : np.linalg.norm(x-y)
        else:
            assert callable(distance)

        D = {}
        D['nodes'] = self.nodes
        D['type'] = 'features'
        D['complete'] = True
        D['edges'] = int(self.nodes*(self.nodes-1)/2)
        D['features'] = features
        D['dfunction'] = lambda i,j :\
            distance(D['features'][i],D['features'][j])
        D['label'] = label
        D['ncolor'] = D['features'][:,0]
        
        self.add_weights(D,**kwargs)
        self.D.append(D)
        self.attributes += 1

    def from_graph(self,elist,dlist,nodes=None,label=None,**kwargs):
        """\
        Adds an attribute to self using lists of edges and distances.
        
        Parameters:
        
        elist : (NN x 2) array-like
        List of edges in the graph.

        dlist : (lenght NN) array-like
        Distances/dissimilarities corresponding to elist.
        """
        assert len(elist) == len(dlist)

        D = {}
        D['nodes'] = self.nodes
        D['type'] = 'graph'
        d['complete'] = False
        D['edges'] = len(elist)
        D['elist'] = elist
        D['dlist'] = dlist
        D['label'] = label
        D['ncolor'] = self.ncolor
        
        self.add_weights(D,**kwargs)
        self.D.append(D)
        self.attributes += 1

    def from_projections(self,attributes=3,X=None,d1=3,Q=None,**kwargs):
        """\
        Adds attributes from projections.
        """
        assert self.attributes == 0
        if X is None:
            X = misc.disk(self.nodes,dim=d1)
        else:
            assert isinstance(X,np.ndarray)
            nodes,dim = X.shape; assert nodes==self.nodes; d1=dim
        if self.ncolor is None:
            self.ncolor = X[:,0]
        proj = projections.PROJ(d1=d1,**kwargs)
        if Q is None or isinstance(Q,str):
            Q = proj.generate(number=attributes,**kwargs)
        else:
            assert len(Q) == attributes
        for k in range(attributes):
            Y = proj.project(Q[k],X)
            self.from_features(Y,**kwargs)

    def compose_distances(self,attribute,function=None,**kwargs):
        if function is None:
            return
        if isinstance(function,str):
            assert function in ['ones','reciprocal']
        if function == 'ones':
            if 'dfunction' in self.D[attribute]:
                self.D[attribute]['dfunction'] = lambda i,j : 1
            if 'dlist' in self.D[attribute]:
                NN = self.D[attribute]['edges']
                self.D[attribute]['dlist'] = np.ones(NN)
        elif function == 'reciprocal':
            if 'dfunction' in self.D[attribute]:
                return None ####
            if 'dlist' in self.D[attribute]:
                dlist = self.D[attribute]['dlist']
                self.D[attribute]['dlist'] = 1.0/dlist
        else:   
            if 'dfunction' in self.D[attribute]:
                return None ####
            if 'dlist' in self.D[attribute]:
                dlist = self.D[attribute]['dlist']
                NN = self.D[attribute]['edges']
                new_dlist = np.empty(NN)
                for i in range(NN):
                    new_dlist[i] = function(dlist[i])
                self.D[attribute]['dlist'] = new_list
                
    def reduce_to_subgraph(self,attribute,**kwargs):
        return None

    ### Sample multigraph ###

    def sample(self,**kwargs):
        Ds = []
        for i in range(self.attributes):
            Ds.append(attribute_sample(self.D[i],**kwargs))
        return Ds
               
    ### Combine attribute ###

    def combine_attributes(self,complete=True,**kwargs):
        """\
        Produces a single attribute that best represents all of the data.
        """
        D0 = {}
        D0['nodes'] = self.nodes
        if complete is True:
            D0['type'] = 'features'
            D0['complete'] = True
            D0['edges'] = int(self.nodes*(self.nodes-1)/2)               
            D0['label'] = 'combined'
            def dfunction(i,j):
                Dij = 0
                for k in range(self.attributes):
                    Dij += self.D[k]['dfunction'](i,j)
                return Dij
            D0['dfunction'] = dfunction
        self.add_weights(D0)
        self.D0 = D0
                
    ### Weights ###

    def add_weights(self,D,weights=None,**kwargs):
        if weights is None:
            D['weighted'] = False
        else:
            D['weighted'] = True
            if isinstance(weights,np.ndarray):
                D['weight_matrix'] = weights
                D['weights'] = lambda i,j : D['weight_matrix'][i,j]
            elif callable(weights):
                D['weight_function'] = weights
                D['weights'] = lambda i,j :\
                    D['weight_function'](D['weight_matrix'][i,j])
            else:
                sys.error('Incorrect weights type')

### Edit graph ###

def remove_edges(D, edge_min_distance=None, edge_max_distance=None,
                 edge_remove_probability=None, **kwargs):
    """\
    Removes edges from graph dictionary as specified.
    
    Parameters:

    D : dictionary
    Graph dictionary.

    min_distance : boolean
    If set to True, removes edges whose distance is less than or equal to
    min_distance.

    max_distance : None or number
    If set to a number, removes edges whose distance is greater than 
    max_distance.

    edge_probability : None or number
    If set to a number, removes edges with probability given by
    edge_probability.
    """
    if edge_min_distance is None and edge_max_distance is None and \
       edge_remove_probability is None:
        print('hi')
        return D
    
    edges = D['edges']; distances = D['distances']; weights = D['weights']
    
    if edge_min_distance is not None:
        keep_indices = []
        for i in range(len(edges)):
            if distances[i] > edge_min_distance:
                keep_indices.append(i)
        edges = edges[keep_indices]
        distances = distances[keep_indices]
        weights = weights[keep_indices]

    if edge_max_distance is not None:
        keep_indices = []
        for i in range(len(edges)):
            if distances[i] < edge_max_distance:
                keep_indices.append(i)
        edges = edges[keep_indices]
        distances = distances[keep_indices]
        weights = weights[keep_indices]

    if edge_remove_probability is not None:
        keep_indices = []
        for i in range(len(edges)):
            if np.random.rand() <= edge_remove_probability:
                keep_indices.append(i)
        edges = edges[keep_indices]
        distances = distances[keep_indices]
        weights = weights[keep_indices]

    D['edges'] = edges
    D['distances'] = distances
    D['weights'] = weights

    return D

### Generate graph ###

def graph_from_coordinates(X,norm=2,edges=None,weights=None,colors=None,
                           **kwargs):
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
        'node_number' : N,
        'node_labels' : range(N),
        'edge_number' : len(e),
        'edges' : e,
        'distances' : d,
        'weights' : w,
        'colors' : colors
        }
    return DD

def graph_from_matrix(D,remove_zeros=True,transformation=None,weights=None):
    """\
    Returns diccionary with dissimilarity relations from dissimilarity matrix.
    
    Parameters:

    D : (N,N) array_like
    Matrix containing pairwise distances/dissimilarities/similarites.

    transformation : None or string or callable
    If None, distances are given by d[(i,j)] = D[i,j].
    Otherwise, distances are given by d[(i,j)] = f(D[i,j]) (unless f(D[i,j]) is
    None, in which case the edge is omitted), where f is determined by the
    given transformation.
    If string, options are 'binary' and 'reciprocal'.
    If callable, f = transformation.

    weights : None or 'relative' or function or array_like
    If weights == None, w_ij = 1
    If weights == 'relative', w_ij = 1/D_ij^2
    If callable(weights), w_ij = weights(D_ij)
    If array_like, w_ij = weights[i,j]
    """
    N = len(D); NN = int(N*(N-1)/2)
    e = np.empty((NN,2),dtype=int)
    d = np.empty(NN)

    if transformation is None:
        it = 0
        for i in range(N):
            for j in range(i+1,N):
                if D[i,j] != 0:
                    e[it] = [i,j]
                    d[it] = D[i,j]
                    it += 1
    else:
        if transformation == 'binary':
            def f(x):
                if x == 0:
                    y = None
                else:
                    y = 1
                return y
        elif transformation == 'reciprocal':
            def f(x):
                if x == 0:
                    y = None
                else:
                    y = 1/x
                return y
        else:
            assert callable(transformation)
            f = transformation
        it = 0
        for i in range(N):
            for j in range(i+1,N):
                Dij = f(D[i,j])
                if Dij is not None:
                    e[it] = [i,j]
                    d[it] = Dij
                    it += 1
    e = e[0:it]
    d = d[0:it]
    w = np.ones(len(e))
    
    DD = {
        'node_number' : N,
        'node_label' : range(N),
        'nodes' : range(N),
        'edges' : e,
        'distances' : d,
        'weights' : w
        }
    return DD

### Generate multigraph ###

def multigraph_from_projections(proj,Q,X,**kwargs):
    """\
    Generates list of graphs generated from objects X using projection rule proj
    with parameters Q.

    X : numpy array
    Positions of objects

    persp : perspective object
    Describes perspectives on X.
    """
    DD = {}
    DD['attribute_number'] = len(Q)
    DD['attribute_labels'] = range(len(Q))
    DD['node_number'] = len(X)
    DD['node_labels'] = range(len(X))
    Y = proj.project(Q,X)
    for k in range(len(Q)):
        D = graph_from_coordinates(Y[k],**kwargs)
        D = remove_edges(D,**kwargs)
        DD[k] = D
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

def remove_edges0(D,number=None,proportion=0.2):
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

def combine(DD,method='maximum'):
    """\
    Combine dissimilarity matrices.
    """
    N = len(DD['nodes'])
    K = len(DD['attributes'])
    D = np.zeros((N,N))
    for k in range(K):
        for edge,distance in zip(DD[k]['edges'],DD[k]['distances']):
            D[edge] = max(D[edge[0],edge[1]],distance)
    D = from_matrix(D)
    return D

### GENERATORS ###

def generate_physical(N,dim=3):
    """\
    Generates a dissimilarity graph from the distances of coordinates.
    """
    X = misc.disk(N,dim=dim)
    D = from_coordinates(X)
    return D

def binomial(N,p,distances=None,K=1):
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

    K : int > 0
    Number of graphs.
    """
    assert isinstance(p,float); assert 0<p<=1
    D = {}
    for k in range(K):
        edges = []
        for i in range(N):
            for j in range(i+1,N):
                if np.random.rand() <= p:
                    edges.append((i,j))
        edges = np.array(edges)
        if distances is None:
            dist = np.ones(len(edges))
        elif distances == 'random':
            dist = np.random.rand(len(edges))
        d = {
            'nodes' : range(N),
            'edges' : edges,
            'distances' : dist
        }
        D[k] = d
    D['attributes'] = range(K)
    D['nodes'] = range(N)
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

class MultiGraph(object):
    """\
    Class of multigraphs to be used in MPSE.
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
