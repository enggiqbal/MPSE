### tSNE implementation ###
import numbers
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import distance
MACHINE_EPSILON = np.finfo(np.double).eps

import misc, gd, plots

def joint_probabilities(distances, perplexity):
    """\
    Computes the joint probabilities p_ij from distances D.

    Parameters
    ----------

    distances : array, shape (N x N) or (N*(N-1)/2,)
    Distances (as a square matrix or unraveled array).

    perpelxity : float, >0
    Desired perpelxity of the joint probability distribution.
    
    Returns
    -------

    P : array, shape (N*(N-1)/2),)
    Condensed joint probability matrix.
    """
    assert isinstance(distances,np.ndarray)
    if len(distances.shape)==1:
        distances = distance.squareform(distances)
    else:
        assert len(distances.shape)==2
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
            P_i /= np.sum(P_i)
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
    P = np.maximum(distance.squareform(P)/sum_P, MACHINE_EPSILON)
    
    return P

### Functions to compute Q given embedded points Y ###

def compute_Q(Y):
    """\
    Compute Q distribution from embeddeded points Y
    """
    N = len(Y)

    dist = distance.pdist(Y,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist/(2.0*np.sum(dist)), MACHINE_EPSILON)
    Q /= np.sum(Q)
    return Q


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
    dist = distance.pdist(embedding,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist/(2.0*np.sum(dist)), MACHINE_EPSILON)
    Q /= np.sum(Q)

    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
        
    return kl_divergence

def gradKL0(P,Q,Y):
    """\
    Gradient of KL diverngence KL(P||Q(Y)) with respect to Y.
    
    Note: see formula (5) in tSNE paper.
    """
    P = distance.squareform(P)
    Q = distance.squareform(Q)
    
    (N,dimY) = Y.shape
    gradient = np.zeros((N,dimY))
    for n in range(N):
        for j in range(N):
            gradient[n] += (P[n,j]-Q[n,j])*(Y[n]-Y[j])*(1+np.sum((Y[n]-Y[j])**2))**(-1)
        gradient[n] *= 4
    return gradient

def grad_KL(P,X):
    
    dist = distance.pdist(X,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist/(2.0*np.sum(dist)), MACHINE_EPSILON)

    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    grad = np.ndarray(X.shape)
    PQd = distance.squareform((P-Q)*dist)
    for i in range(len(X)):
        grad[i] = np.dot(np.ravel(PQd[i],order='K'),X[i]-X)
    grad *= 4
    
    return (kl_divergence,grad)

class TSNE(object):
    """\
    Class to solve tsne problems
    """
    def __init__(self, D, dim=2, perplexity=30.0, verbose=0,
                 indent='', title='', **kwargs):
        """\
        Initializes TSNE object
        """

        self.verbose = verbose; self.title = title; self.indent = indent
        
        assert isinstance(D,np.ndarray)
        self.N = len(D); assert D.shape == (self.N,self.N)
        self.D = D

        assert isinstance(dim,int); assert dim > 0
        self.dim = dim

        self.P = joint_probabilities(self.D,perplexity)

        self.f = lambda X: KL(self.P,X)
        self.F = lambda X: grad_KL(self.P,X)

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

        print(self.perplexity)

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
            
        self.X = X0
        self.update(**kwargs)
        self.X0 = self.X.copy()
        
        if self.verbose > 0:
            print(f'    initial cost : {self.cost:0.2e}')


    def update(self):
        self.cost = KL(self.P,self.X)

    def gd(self, scheme='mm', verbose=0, **kwargs):
        if hasattr(self,'X') is False:
            self.initialize(title='automatic',**kwargs)
        if self.verbose > 0:
            print(self.indent+f'  MDS.gd({self.title}):')

        F = lambda X : self.F(X)
        self.X, H = gd.single(self.X,F,scheme=scheme,
                              verbose=self.verbose,
                              indent=self.indent+'    ',
                              **kwargs)
        self.update()
        if self.verbose > 0:
            print(self.indent+f'    final stress : {self.cost:0.2e}')
            
    def figureX(self,title='',edges=False,node_color=None,labels=None,
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
        #if node_color is None:
        #    node_color = self.D['node_colors']
        plots.plot2D(self.X,edges=edges,colors=node_color,labels=labels,
                     axis=axis,ax=ax,title=title,**kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)


### TESTS ###

def example_tsne():
    X_true = np.load('examples/123/true2.npy')#[0:500]
    from scipy import spatial
    D = spatial.distance_matrix(X_true,X_true)

    vis = TSNE(D,verbose=2,perplexity=30)
    vis.initialize(X0=X_true)
    vis.figureX()
    vis.gd(scheme='fixed',lr=100,verbose=3,plot=True)
    vis.figureX()
    plt.show()
    #mv = mpse.Multiview(D,dimX=2)
    #mv.setup_technique('tsne',perplexity=2)
    #mv.initialize_X()
    #mv.figureX(); plt.show()
    #stress0 = mv.cost
    #X0 = mv.X
    #mv.solve_X(algorithm='gd',rate=.5,max_iters=50)
    #mv.figureX(); plt.show()
    #X = mv.X; stress = mv.cost

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
    
