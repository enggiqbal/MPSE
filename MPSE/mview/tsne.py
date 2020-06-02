### tSNE implementation ###
import numbers
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import distance
MACHINE_EPSILON = np.finfo(np.double).eps

import misc, gd, plots

def compute_Pi(D,i,sigmai):
    """\
    Vector with conditional probabilities of p for entries 1 to n with respect
    to datapoint i.
    
    --- arguments ---
    D = Distance matrix
    i = index
    sigmai = variance of Gaussian centered on datapoint i

    Note: see equation (1) in tSNE paper
    """
    Pi = np.exp(-D[i]**2/(2*sigmai**2))
    Pi[i] = 0
    Pi /= np.sum(Pi)
    return Pi

def compute_PerpPi(Pi,i):
    """\
    Perplexity of node i given current conditional probability Pi
    """
    Pi0 = np.delete(Pi,i)
    HPi = -np.dot(Pi0,np.log2(Pi0+1e-32)) ###
    PerpPi = 2**(HPi)
    return PerpPi

def find_sigmai(D,i,perplexity):
    """\
    Finds variance of Gaussian centered on datapoint i that produces the
    desired perplexity.

    --- arguments ---
    D = distance matrix
    i = index
    perplexity = target perpelxity

    Note: see page (4) of tSNE paper
    """
    lower_bound=1e-2; upper_bound=1e2; iters=10 #parameters for binary search
    
    for iter in range(iters):
        sigmai = (lower_bound*upper_bound)**(1/2)
        Pi = compute_Pi(D,i,sigmai); 
        PerpPi = compute_PerpPi(Pi,i)
        if PerpPi > perplexity:
            upper_bound = sigmai
        else:
            lower_bound = sigmai            
    return (lower_bound*upper_bound)**(1/2)

def find_sigma(D,perplexity):
    N = len(D)
    sigma = np.empty(N)
    for n in range(N):
        sigma[n] = find_sigmai(D,n,perplexity)
    return sigma

def compute_P(D,sigma):
    N = len(D)
    P = np.empty((N,N))
    for n in range(N):
        P[n] = compute_Pi(D,n,sigma[n])
    P = (P+P.T)#/(2*N)
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
    Q = distance.squareform(Q)
    return Q


### Cost function and gradient ###

def KL(P,Q):
    """\
    KL divergence KL(P||Q) between distributions P and Q.
    It is assumed that P and Q are symmetric with zero diagonals.
    """
    assert isinstance(P,np.ndarray)
    assert isinstance(Q,np.ndarray)
    N = len(P)
    C = 0
    for i in range(N):
        for j in [k for k in range(N) if k != i]:
            C -= P[i,j]*np.log(P[i,j]/Q[i,j]+1e-32) ###
    return C

def gradKL(P,Q,Y):
    """\
    Gradient of KL diverngence KL(P||Q(Y)) with respect to Y.
    
    Note: see formula (5) in tSNE paper.
    """
    assert isinstance(P,np.ndarray) and isinstance(Q,np.ndarray)
    assert isinstance(Y,np.ndarray)
    
    (N,dimY) = Y.shape
    gradient = np.zeros((N,dimY))
    for n in range(N):
        for j in range(N):
            gradient[n] += (P[n,j]-Q[n,j])*(Y[n]-Y[j])*(1+np.sum((Y[n]-Y[j])**2))**(-1)
        gradient[n] *= 4
    return gradient

def F(X,P):
    Q = compute_Q(X)
    cost = KL(P,Q)
    grad = gradKL(P,Q,X)
    return (cost,grad)

class TSNE(object):
    """\
    Class to solve tsne problems
    """
    def __init__(self, D, dim=2, sigma='optimal', perplexity=30.0, verbose=0,
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

        self.set_sigma(sigma,perplexity)
        self.P = compute_P(self.D,self.sigma)

        self.f = lambda Y: KL(self.P,compute_Q(Y))
        self.df = lambda Y: gradKL(self.P,compute_Q(Y),Y)
        self.F = lambda Y: F(Y,self.P)

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
        self.Q = compute_Q(self.X)
        self.cost = KL(self.P,self.Q)
        self.gradient = gradKL(self.P,self.Q,self.X)

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

    from sklearn.manifold import TSNE as tsne
    X_embedded = tsne(n_components=2,verbose=2,method='exact').fit_transform(X_true)
    plt.figure()
    plt.plot(X_embedded[:,0],X_embedded[:,1],'o')
    plt.show()

    vis = TSNE(D,verbose=2,perplexity=30)
    vis.initialize(X0=X_true)
    vis.figureX()
    vis.gd(scheme='fixed',lr=0.1,verbose=3,plot=True)
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
    
if __name__=='__main__':
    print('mview.tsne : tests')
    example_tsne()
    
