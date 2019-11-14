### tSNE implementation ###
import numbers
import numpy as np

class TSNE(object):
    """\
    Class to solve tsne problems
    """
    def __init__(self, D, dim, sigma='optimal', perplexity=30.0):
        
        assert isinstance(D,np.ndarray)
        self.N = len(D); assert D.shape == (self.N,self.N)
        self.D = D

        assert isinstance(dim,int); assert dim > 0
        self.dim = dim

        self.set_sigma(sigma,perplexity)
        self.P = compute_P(self.D,self.sigma)

        self.f = lambda Y: KL(self.P,compute_Q(Y))
        self.df = lambda Y: gradKL(self.P,compute_Q(Y),Y)

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

    def update(self,Y):
        self.Y = Y
        self.Q = compute_Q(Y)
        self.cost = KL(self.P,self.Q)
        self.gradient = gradKL(self.P,self.Q,Y)


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
    P = (P+P.T)/(2*N)
    return P

### Functions to compute Q given embedded points Y ###

def compute_Q(Y):
    """\
    Compute Q distribution from embeddeded points Y
    """
    N = len(Y)
    Q = np.empty((N,N))
    for n in range(N):
        Qn = (1+np.sum((Y-Y[n])**2,1))**(-1)
        Qn[n] = 0
        Qn /= np.sum(Qn)
        Q[n] = Qn
    #Q = (Q+Q.T)/2
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
            C += P[i,j]*np.log(P[i,j]/Q[i,j]+1e32) ###
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
