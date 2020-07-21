### tSNE implementation ###
import numbers
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial.distance
MACHINE_EPSILON = np.finfo(np.double).eps

import misc, gd, plots, setup

def joint_probabilities(distances, perplexity):
    """\
    Computes the joint probabilities p_ij from distances D.

    Parameters
    ----------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Condensed distances.

    perpelxity : float, >0
    Desired perpelxity of the joint probability distribution.
    
    Returns
    -------

    P : array, shape (N*(N-1)/2),)
    Condensed joint probability matrix.
    """
    distances = scipy.spatial.distance.squareform(distances)
    n_samples = len(distances)
    #Find optimal neighborhood parameters to achieve desired perplexity
    lower_bound=1e-2; upper_bound=1e2; iters=10 #parameters for binary search
    sigma = np.empty(n_samples) #bandwith array
    for i in range(n_samples):
        #initialize bandwith parameter for sample i:
        sigma_i = (lower_bound*upper_bound)**(1/2)
        for iter in range(iters):
            #distances to sample i, not including self:
            D_i = np.delete(distances[i],i) 
            #compute array with conditional probabilities w.r.t. sample i:
            P_i = np.exp(-D_i**2/(2*sigma_i**2))
            P_i /= np.sum(P_i) ####
            #compute perplexity w.r.t sample i:
            HP_i = -np.dot(P_i,np.log2(P_i+MACHINE_EPSILON))
            PerpP_i = 2**(HP_i)
            #update bandwith parameter for sample i:
            if PerpP_i > perplexity:
                upper_bound = sigma_i
            else:
                lower_bound = sigma_i
        #final bandwith parameter for sample i:
        sigma[i] = (lower_bound*upper_bound)**(1/2)

    conditional_P = np.exp(-distances**2/(2*sigma**2))
    np.fill_diagonal(conditional_P,0)
    conditional_P /= np.sum(conditional_P,axis=1)
    
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(scipy.spatial.distance.squareform(P)/sum_P, MACHINE_EPSILON)
    return P

### Cost function and gradient ###

def KL(P,embedding):
    """\
    KL divergence KL(P||Q) between distributions P and Q, where Q is computed
    from the student-t distribution from the given embedding array.

    Parameters
    ----------

    P : array, shape (n_samples*(n_samples-1)/2,)
    Condensed probability array.
    
    embedding : array, shape (n_samples,dim)
    Current embedding.

    Results
    -------

    kl_divergence : float
    KL-divergence KL(P||Q).
    """
    # compute Q:
    dist = scipy.spatial.distance.pdist(embedding,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist/(2.0*np.sum(dist)), MACHINE_EPSILON)
    
    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
        
    return kl_divergence

def grad_KL(P,embedding,only_gradient=False):
    """\
    Computes KL divergence and its gradient at the given embedding.

    Parameters
    ----------

    P : array, shape (n_samples*(n_samples-1)/2,)
    Condensed probability array.
    
    embedding : array, shape (n_samples,dim)
    Current embedding.

    Results
    -------

    kl_divergence : float
    KL-divergence KL(P||Q).

    grad : float
    gradiet of KL(P||Q(X)) w.r.t. X.
    """
    dist = scipy.spatial.distance.pdist(embedding,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist/(2.0*np.sum(dist)), MACHINE_EPSILON) ######
    
    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    grad = np.ndarray(embedding.shape)
    PQd = scipy.spatial.distance.squareform((P-Q)*dist)
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(PQd[i],order='K'),embedding[i]-embedding)
    grad *= 4
    
    return grad, kl_divergence

def batch_gradient(P, embedding, batch_size=10, indices=None, weights=None,
                   return_objective=True):
    """\
    Returns gradient approximation.
    """
    n_samples = len(embedding)
    if indices is None:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    else:
        assert len(indices) == n_samples
    grad = np.empty(embedding.shape)
    stress = 0
    for start in range(0, n_samples, batch_size):
        end = min(start+batch_size,n_samples)
        batch_idx = np.sort(indices[start:end])
        embedding_batch = embedding[batch_idx]
        P_batch = P[setup.batch_indices(batch_idx,n_samples)]
        grad[batch_idx], st0 = grad_KL(P_batch,
                                       embedding_batch)
        stress += st0

    return grad, stress

class TSNE(object):
    """\
    Class to solve tsne problems
    """
    def __init__(self, data, dim=2, perplexity=30.0, sample_colors=None,
                 verbose=0,
                 indent='', title='', **kwargs):
        """\
        Initializes TSNE object.

        Parameters
        ----------

        data : array or dictionary
        Contains distances or dissimilarities among a set of objects.
        Can be either of the following:

        i) array, shape (N x N)
        Distance/dissimilarity matrix
        
        ii) array, shape (N x dim)
        Positions/featurs

        iii) dictionary
        See dissimilarities.py

        dim : int > 0
        Embedding dimension.

        perplexity : float > 0
        Perplexity used in determining the conditional probabilities p(i|j).
        """
        if verbose > 0:
            print(indent+'mview.TSNE():')

        self.sample_colors = sample_colors
        self.verbose = verbose; self.title = title; self.indent = indent
        
        self.distances = setup.setup_distances(data)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances)
        self.N = self.n_samples
        self.D = self.distances

        assert isinstance(dim,int); assert dim > 0
        self.dim = dim

        if verbose > 0:
            print(indent+'  data details:')
            print(indent+f'    number of samples : {self.n_samples}')
            print(indent+'  embedding details:')
            print(indent+f'    embedding dimension : {dim}')
            print(indent+f'    perplexity : {perplexity:0.2f}')

        self.P = joint_probabilities(self.D,perplexity)

        self.objective = lambda X, P=self.P, **kwargs : KL(P,X)
        def gradient(embedding,batch_size=None,indices=None,**kwargs):
            if batch_size is None or batch_size >= self.n_samples:
                return grad_KL(self.P,embedding)
            else:
                return batch_gradient(self.P,embedding,batch_size,indices)
        self.gradient = gradient

        self.initialize()

    def set_sigma(self,sigma='optimal',perplexity=30.0):
        if isinstance(sigma,numbers.Number):
            assert sigma > 0
            self.sigma = np.ones(self.N)*sigma
        elif isinstance(sigma,np.ndarray):
            assert sigma.shape == (self.N,)
            assert all(sigma>0)
            self.sigma = sigma
        else:
            assert sigma is 'optimal'
            assert isinstance(perplexity,numbers.Number)
            assert perplexity > 0
            self.perplexity = perplexity
            self.sigma = find_sigma(self.D,self.perplexity)

    def initialize(self, X0=None, **kwargs):
        """\
        Set initial embedding.
        """
        if self.verbose > 0:
            print(f'  MDS.initialize({self.title}):')
            
        if X0 is None:
            X0 = misc.initial_embedding(self.N,dim=self.dim,
                                        radius=1,**kwargs)
                                        #radius=self.D['rms'],**kwargs)
            if self.verbose > 0:
                print('    method : random')
        else:
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.N,self.dim)
            if self.verbose > 0:
                print('    method : initialization given')
            
        self.embedding = X0
        self.update(**kwargs)
        self.embedding0 = self.embedding.copy()
        
        if self.verbose > 0:
            print(f'    initial cost : {self.cost:0.2e}')


    def update(self,**kwargs):
        self.cost = self.objective(self.embedding)

    def gd(self, batch_size=None, **kwargs):
        if self.verbose > 0:
            print(self.indent+'  TSNE.gd():')
            print(self.indent+'    specs:')

        if batch_size is None or batch_size >= self.n_samples:
            Xi = None
            F = lambda embedding : self.gradient(embedding)
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
            F = lambda X, indices : self.gradient(X,batch_size,indices)
            if self.verbose > 0:
                print(self.indent+'      gradient type : batch')
                print(self.indent+'      batch size :',batch_size)

        self.embedding, H = gd.single(self.embedding,F,Xi=Xi,
                              verbose=self.verbose,
                              indent=self.indent+'    ',
                              **kwargs)
        self.update()
        if self.verbose > 0:
            print(self.indent+f'    final stress : {self.cost:0.2e}')
            
    def plot_embedding(self,title='',edges=False,colors='default',labels=None,
                axis=True,plot=True,ax=None,**kwargs):
        assert self.dim >= 2
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        if edges is True:
            edges = self.D['edge_list']
        elif edges is False:
            edges = None
        if colors == 'default':
            colors = self.sample_colors
        plots.plot2D(self.embedding,edges=edges,colors=colors,labels=labels,
                     axis=axis,ax=ax,title=title,**kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)


### TESTS ###

def example_tsne(**kwargs):
    X_true = np.load('examples/123/true2.npy')#[0:500]
    colors = misc.labels(X_true)
    from scipy import spatial
    D = spatial.distance_matrix(X_true,X_true)

    vis = TSNE(D,verbose=2,perplexity=50,sample_colors=colors)
    vis.initialize(X0=X_true)
    vis.plot_embedding()
    vis.gd(plot=True,**kwargs)
    vis.plot_embedding()
    plt.show()

def sk_tsne():

    X_true = np.load('examples/123/true2.npy')#[0:500]
    from scipy import spatial
    D = spatial.distance_matrix(X_true,X_true)
    
    from sklearn.manifold import TSNE as tsne
    X_embedded = tsne(n_components=2,verbose=2,method='exact').fit_transform(X_true)
    plt.figure()
    plt.plot(X_embedded[:,0],X_embedded[:,1],'o')
    plt.show()
    
if __name__=='__main__':
    print('mview.tsne : tests')
    example_tsne()
    
