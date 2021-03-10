### MDS implementation ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
import scipy.spatial.distance

import misc, setup, multigraph, gd, plots

### MDS STRESS AND GRADIENT FUNCTIONS ###

def stress(distances, embedding, weights=None, normalize=True):
    """\
    Returns MDS stress for given set of distances and embedding.

    Parameters
    ----------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Condensed true distances.

    embedding : array, shape (n_samples, embedding_dimension)
    Embedding.

    weights : array, shape (n_samples*(n_samples-1)/2,)
    MDS weights. If None given, the MDS stress function is unweighted.

    normalize : boolean
    If set to True, returns normalized stress.

    Return
    ------

    stress : float
    Stress for given distances and embedding.
    """
    dist = scipy.spatial.distance.pdist(embedding) #embedding distances
    diff = distances-dist
    if weights is None:
        stress = np.linalg.norm(diff)**2
        if normalize:
            stress = math.sqrt(stress/len(distances))
    else:
        stress = np.dot(weights,diff**2)
        if normalize:
            stress = math.sqrt(stress/np.sum(weights))
    return stress

def full_gradient(distances, embedding, weights=None, normalize=True,
                  minimum_distance=None, return_objective=True):
    """\
    Returns gradient of MDS stress function, along with MDS stress function
    value (optional).

    Parameters
    ----------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Condensed distances.

    embedding : array, shape (n_samples, embedding_dimension)
    Embedding.

    weights : array, shape (n_samples*(n_samples-1)/2,)
    Array containing weights of samples pairs, used in determining contribution
    of each pair to MDS stress. If None given, then the MDS stress function is
    unweighted.

    normalize : boolean
    If set to True, returns normalized stress.

    minimum_distance : float or None
    Minimum distance allowed in embedding.

    Return
    ------

    grad : array, shape (n_samples,embedding_dimension)
    Gradient of MDS stress function for given distances and embedding.

    stress : float
    Stress (or estimate) for given embedding. This is returned by default, but
    can be suppresed by setting return_objective to False.
    """
    #stability parameters:
    min_dist = 1e-6 #minimum embedding distance used
    
    grad = np.zeros(embedding.shape)
    stress = 0
    dist = scipy.spatial.distance.pdist(embedding)
    if minimum_distance is not None:
        dist = np.maximum(minimum_distance,dist)
    diff = dist-distances

    if weights is None:
        constants = 2*diff/dist
    else:
        constants = 2*weights*diff/dist

    grad_terms = scipy.spatial.distance.squareform(constants)
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(grad_terms[i],order='K'),
                                  embedding[i]-embedding)

    if normalize:
        if weights is None:
            grad /= np.linalg.norm(distances)
        else:
            grad /= np.sqrt(np.dot(weights,distances**2))

    if return_objective:
        if weights is None:
            stress = np.linalg.norm(diff)**2
        else:
            stress = np.dot(weights,diff**2)
        if normalize:
            if weights is None:
                stress = math.sqrt(stress/len(distances))
            else:
                stress = math.sqrt(stress/np.sum(weights))

    if return_objective:
        return grad, stress
    else:
        return grad

def batch_gradient(distances, embedding, batch_size=10, indices=None,
                   weights=None, normalize=True, minimum_distance=None,
                   return_objective=True):
    """\
    Returns gradient of MDS stress function for given batch, along with the
    corrresponding portion of the MDS stress function value (optional).

    Parameters
    ----------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Condensed distances.

    embedding : array, shape (n_samples, embedding_dimension)
    Embedding.

    batch_indices : list
    Indices of samples to be used.

    weights : array, shape (n_samples*(n_samples-1)/2,)
    Array containing weights of samples pairs, used in determining contribution
    of each pair to MDS stress. If None given, then the MDS stress function is
    unweighted.

    normalize : boolean
    If set to True, returns normalized stress.

    Return
    ------

    grad : array, shape (n_samples,embedding_dimension)
    Gradient of MDS stress function for given distances and embedding.

    stress : float
    Stress (or estimate) for given embedding. This is returned by default, but
    can be suppresed by setting return_objective to False.
    """
    n_samples = len(embedding)
    if indices is None:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    else:
        assert len(indices) == n_samples
    grad = np.empty(embedding.shape)
    stress = 0
    weights_batch = None
    for start in range(0, n_samples, batch_size):
        end = min(start+batch_size,n_samples)
        batch_idx = np.sort(indices[start:end])
        embedding_batch = embedding[batch_idx]
        batch_indices = setup.batch_indices(batch_idx,n_samples)
        distances_batch = distances[batch_indices]
        if weights is not None:
            weights_batch = weights[batch_indices]
        grad[batch_idx], st0 = full_gradient(distances_batch, embedding_batch,
                                             weights=weights_batch,
                                             minimum_distance=minimum_distance)
        stress += st0**2
    if normalize:
        n_batches = math.ceil(n_samples/batch_size)
        grad /= np.linalg.norm(distances)/n_samples*batch_size
        stress = math.sqrt(stress/n_batches)

    if return_objective:
        return grad, stress
    else:
        return grad
        

def stress_function(X,D,normalize=True,estimate=True,weighted=False,**kwargs):
    """\

    Normalized MDS stress function.

    Parameters:

    X : numpy array
    Position/coordinate/embedding array.

    D : numpy array or dictionary
    Either a dissimilarity matrix or a dissimilarity dictionary as specified
    in mview.multigraph.

    normalize : boolean
    If set to True, returns normalized stress.

    estimate : boolean
    If set to True, it estimates stress to reduce computation time.

    weighted : boolean
    If set to True, uses given weights. As of now, for dictionaries only.

    Returns:

    stress : float
    MDS stress at X.
    """
    estimate_default = 128 #default average number of edges used in estimation
    if isinstance(D,np.ndarray):
        if estimate is False:
            dX = sp.spatial.distance_matrix(X,X)
            stress = np.linalg.norm(D-dX)
            if normalize is True:
                stress /= np.linalg.norm(D)
            else:
                stress = (stress/2)**2
        else:
            N = len(D)
            if estimate is True:
                estimate = min(estimate_default,N)
            edges = misc.random_triangular(N,int(estimate*(estimate-1)/2))
            stress = 0
            for i1,i2 in edges:
                dX = np.linalg.norm(X[i1]-X[i2])
                stress += (dX-D[i1,i2])**2
            if normalize is True:
                normalization_factor = 0
                for i1,i2 in edges:
                    normalization_factor += D[i1,i2]**2
                stress = math.sqrt(stress/normalization_factor)
    else:
        assert isinstance(D,dict)
        if estimate is False:
            if D['type'] == 'matrix':
                dX = scipy.spatial.distance_matrix(X,X)
                stress = np.linalg.norm(D['matrix']-dX)
                stress /= math.sqrt(D['node_number']*(D['node_number']-1)/2)* \
                    D['rms']
            elif D['complete'] is True:
                stress = 0
                for i in range(D['node_number']):
                    for j in range(D['node_number']):
                        dXij = np.linalg.norm(X[i]-X[j])
                        stress += (D['dfunction'](i,j)-dXij)**2
                stress /= D['node_number']*(D['node_number']-1)/2
                stress = math.sqrt(stress)/D['rms']
            else:
                stress = 0
                if weighted is False:
                    for i in range(D['edge_number']):
                        i1,i2 = D['edge_list'][i]
                        dXij = np.linalg.norm(X[i1]-X[i2])
                        stress += (D['dissimilarity_list'][i]-dXij)**2
                else:
                    assert weighted is True
                    assert 'weights' in D
                    for i in range(D['edge_number']):
                        i1,i2 = D['edge_list'][i]
                        weight = D['weights'][i]
                        dXij = np.linalg.norm(X[i1]-X[i2])
                        dist = D['dissimilarity_list'][i]
                        stress += weight*((dist-dXij)**2)
                stress = math.sqrt(stress/D['edge_number'])/D['rms']
        else:
            if estimate is True:
                estimate = estimate_default
            edge_number = min(int(estimate*(estimate-1)/2),D['edge_number'])
            stress = 0
            if D['complete'] is True:
                edges = misc.random_triangular(D['node_number'],edge_number)
                for i1,i2 in edges:
                    dX = np.linalg.norm(X[i1]-X[i2])
                    stress += (dX-D['dfunction'](i1,i2))**2
            else:
                inds = np.random.choice(D['edge_number'],edge_number,
                                        replace=False)
                if weighted is False:
                    for i in range(edge_number):
                        i1,i2 = D['edge_list'][inds[i]]
                        dX = np.linalg.norm(X[i1]-X[i2])
                        stress += (dX-D['dissimilarity_list'][inds[i]])**2
                else:
                    assert weighted is True
                    assert 'weights' in D
                    for i in range(edge_number):
                        i1,i2 = D['edge_list'][inds[i]]
                        dX = np.linalg.norm(X[i1]-X[i2])
                        weight = D['weights'][inds[i]]
                        dist = D['dissimilarity_list'][inds[i]]
                        stress += weight*((dist-dX)**2)
            stress = math.sqrt(stress/edge_number)/D['rms']
    return stress

def F(X,D,normalize=True,weighted=False):
    """\
    Returns exact stress and gradient for embedding X with target distances D.

    Parameters:

    X : numpy array
    Position/coordinate/embedding array.

    D : numpy array or dictionary
    Either a dissimilarity matrix or a dissimilarity dictionary as specified
    in mview.multigraph.

    normalize : boolean
    If set to True, returns normalized stress and gradient.

    Returns:

    stress : float
    MDS stress at X.

    gradient : numpy array
    MDS stress gradient at X. 
    """    
    N, dim = X.shape
    if D['complete'] is True:
        assert weighted is False
        fX = 0
        dfX = np.zeros(X.shape)
        for i in range(N):
            for j in range(i+1,N):
                Xij = X[i]-X[j]
                dij = np.linalg.norm(Xij)
                diffij = dij-D['dfunction'](i,j)
                fX += diffij**2
                dXij = 2*diffij/dij*Xij
                dfX[i] += dXij
                dfX[j] -= dXij
        if normalize is True:
            fX = math.sqrt(fX/((N*(N-1)/2)*D['rms']**2))
            dfX /= math.sqrt(N*(N-1))*D['rms']
    else:
        assert D['type'] == 'graph'
        dfX = np.zeros(X.shape)
        fX = 0
        if weighted is False:
            for i in range(D['edge_number']):
                i1,i2 = D['edge_list'][i]
                Xij = X[i1]-X[i2]
                dij = np.linalg.norm(Xij)
                diffij = dij-D['dissimilarity_list'][i]
                fX += diffij**2
                dXij = 2*diffij/dij*Xij
                dfX[i1] += dXij
                dfX[i2] -= dXij
        else:
            assert weighted is True
            assert 'weights' in D
            for i in range(D['edge_number']):
                i1,i2 = D['edge_list'][i]
                Xij = X[i1]-X[i2]
                dij = np.linalg.norm(Xij)
                weight = D['weights'][i]
                diffij = dij-D['dissimilarity_list'][i]
                fX += weight*diffij**2
                dXij = weight*2*diffij/dij*Xij
                dfX[i1] += dXij
                dfX[i2] -= dXij
        if normalize is True:
            fX = math.sqrt(fX/(D['edge_number']*D['rms']**2))
            dfX /= math.sqrt(2*D['edge_number'])*D['rms']
    return fX, dfX

class MDS(object):
    """\
    Class with methods to solve multi-dimensional scaling problems.
    """
    def __init__(self, data, dim=2, weights=None, estimate=True, safety=1e-4,
                 normalize=True, initial_embedding='random',
                 sample_colors=None, verbose=0, indent='', **kwargs):
        """\
        Initializes MDS object.

        Parameters:

        data : array or dictionary
        Distance/dissimilarity/feature data, which can have any of the 
        following formats:
        1) a 1D condensed distance array
        2) a square distance matrix/array
        3) a feature array
        4) a dictionary describing a graph

        dim : int > 0
        Embedding dimension.

        weights : None or str or callable or array
        Weights to be used in defining MDS stress.

        verbose : int >= 0
        Print status of methods in MDS object if verbose > 0.

        indent : str
        When printing, add indent before printing every new line.
        """
        self.verbose = verbose
        self.indent = indent
        if self.verbose > 0:
            print(self.indent+'mview.MDS():')
            
        self.distances = setup.setup_distances(data, **kwargs)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances)

        if safety is None:
            self.minimum_distance = None
        else:
            assert safety > 0 and safety <= 1e-2
            self.minimum_distance = np.max(self.distances)*safety
            self.distances = np.maximum(self.distances,self.minimum_distance)
            
        self.weights = setup.setup_weights(self.distances, weights=weights)
        self.normalize = normalize
                
        if sample_colors is None:
            self.sample_colors = self.distances[0:self.n_samples]
        else:
            self.sample_colors = sample_colors
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim

        assert isinstance(estimate,bool)
        self.estimate = estimate


        self.objective = lambda X, **kwargs : stress(
            self.distances, X, weights=self.weights, normalize=self.normalize)
        def gradient(embedding, batch_size=None, indices=None, **kwargs):
            if batch_size is None or batch_size >= self.n_samples:
                return full_gradient(
                    self.distances,embedding,
                    weights=self.weights, normalize=self.normalize,
                    minimum_distance=self.minimum_distance)
            else:
                return batch_gradient(
                    self.distances,embedding, batch_size, indices,
                    weights=self.weights, normalize=self.normalize,
                    minimum_distance=self.minimum_distance)
        self.gradient = gradient

        if verbose > 0:
            print(indent+'  data details:')
            print(indent+f'    number of samples : {self.n_samples}')
            if self.weights is None:
                print(indent+f'    weighted : False')
            else:
                print(indent+f'    weighted : True')
            print(indent+'  embedding details:')
            print(indent+f'    embedding dimension : {self.dim}')

        #save or compute initial embedding
        if isinstance(initial_embedding,np.ndarray):
            assert initial_embedding.shape == (self.n_samples,self.dim)
            if self.verbose > 0:
                print('    initial embedding : given')
            self.X0 = initial_embedding
            self.X = self.X0
        elif initial_embedding == 'random':
            self.X0 = misc.initial_embedding(self.n_samples,dim=self.dim,
                                        radius=1,**kwargs)
            self.X = self.X0
            if self.verbose > 0:
                print('    initial embedding : random')
        else:
            assert initial_embedding is None

        #save initial costs
        if initial_embedding is not None:
            self.initial_cost = self.objective(self.X0,**kwargs)
            self.cost = self.initial_cost
            if self.verbose > 0:
                print(f'    initial stress : {self.cost:0.2e}')

        self.computation_history = []

    def update(self,X,H,**kwargs):
        self.X = X
        self.cost = self.objective(self.X,**kwargs)
        self.computation_history.append(H)   

    ### Methods to update MDS embedding ###

    def gd(self, batch_size=None, **kwargs):
        if self.verbose > 0:
            print(self.indent+'  MDS.gd():')
            print(self.indent+'    specs:')

        if batch_size is None or batch_size >= self.n_samples:
            Xi = None
            F = lambda X : self.gradient(X)
            if self.verbose > 0:
                print(self.indent+'      gradient type : full')
        else:
            def Xi():
                indices = np.arange(self.n_samples)
                np.random.shuffle(indices)
                xi = {
                    'indices' : indices
                }
                return xi
            F = lambda X, indices : self.gradient(X,batch_size=batch_size,
                                                  indices=indices)
            if self.verbose > 0:
                print(self.indent+'      gradient type : batch')
                print(self.indent+'      batch size :',batch_size)
        X, H = gd.single(self.X,F,Xi=Xi,
                         verbose=self.verbose,
                         indent=self.indent+'    ',
                         **kwargs)
        self.update(X,H,**kwargs)
        if self.verbose > 0:
            print(self.indent+f'    final stress : {self.cost:0.2e}')

    ### PLOTS GENERATORS ###

    def plot_embedding(self,title='embedding',edges=False,colors='default',
                       labels=None,
                       axis=True,plot=True,
                       ax=None,**kwargs):
        assert self.dim >= 2
        if edges is True:
            edges = self.distances['edge_list']
        elif edges is False:
            edges = None
        if colors == 'default':
            colors = self.sample_colors

        if self.dim == 2:
            plots.plot2D(self.X,edges=edges,colors=colors,labels=labels,
                         axis=axis,ax=ax,title=title,**kwargs)
        else:
            plots.plot3D(self.X,edges=edges,colors=colors,title=title,
                         ax=ax,**kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)

    def plot_computations(self,title='computations',plot=True,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False

        costs = np.array([])
        grads = np.array([])
        lrs = np.array([])
        steps = np.array([])
        iterations=0; markers = []
        for H in self.computation_history:
            if iterations != 0:
                ax.axvline(x=iterations-1,ls='--',c='black',lw=.5)
            iterations += H['iterations']
            costs = np.concatenate((costs,H['costs']))
            grads = np.concatenate((grads,H['grads']))
            lrs = np.concatenate((lrs,H['lrs']))
            steps = np.concatenate((steps,H['steps']))
        ax.semilogy(costs,label='stress',linewidth=3)
        ax.semilogy(grads,label='grad size')
        ax.semilogy(lrs,label='lr')
        ax.semilogy(steps,label='step size')
        ax.legend()
        ax.set_title(title)
        if plot is True:
            plt.draw()
            plt.pause(1.0)
                                   
### TESTS ###

def basic(example='mnist', **kwargs):
    import samples
    if example == 'mnist':
        X, labels, = samples.mnist()
    distances = X
        
    mds = MDS(distances,dim=2,verbose=2,
              sample_colors=labels)

    fig, ax = plt.subplots(1,3,figsize=(9,3))
    fig.suptitle('MDS - disk data')
    fig.subplots_adjust(top=0.80)
    mds.plot_embedding(title='initial embedding',ax=ax[0])
    mds.gd(min_cost=1e-6,batch_size=50,**kwargs)
    mds.gd(batch_size=200)
    mds.plot_computations(ax=ax[1])
    mds.plot_embedding(title='final embedding',ax=ax[2])
    plt.draw()
    plt.pause(1.0)
        
def disk(N=128,weights=None,**kwargs):
    #basic disk example
    #N is number of points
    #weights: use None or 'reciprocal' or array, etc
    
    print('\n***disk example***\n')
    
    X = misc.disk(N,2); colors = misc.labels(X)
    distances = scipy.spatial.distance.pdist(X)
    
    title = 'basic disk example'
        
    mds = MDS(distances,weights=weights,dim=2,verbose=2,title=title,
              sample_colors=colors)

    fig, ax = plt.subplots(1,3,figsize=(9,3))
    fig.suptitle('MDS - disk data')
    fig.subplots_adjust(top=0.80)
    mds.plot_embedding(title='initial embedding',ax=ax[0])
    mds.gd(min_cost=1e-6,**kwargs)
    mds.plot_computations(ax=ax[1])
    mds.plot_embedding(title='final embedding',ax=ax[2])
    plt.draw()
    plt.pause(1.0)

if __name__=='__main__':

    print('mview.mds : running tests')
    N = 100
    #weights = np.ones(N)
    #weights = np.random.rand(100)
    #weights = np.concatenate((np.ones(int(N*0.8)),np.zeros(N-int(N*0.8))))
    #disk(N,batch_size=50,max_iter=100)
    basic()
    plt.show()
    
