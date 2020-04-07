import os, sys
import numpy as np
import scipy.stats
import math, numbers, itertools

families = ['linear']
constraints = [None,'orthogonal','similar']

### LINEAR PROJECTIONS ###

class PROJ(object):
    """\
    Class use to define the desired collection of perspective functions and    
    containing methods to call perspective functions in such collection.
    """
    
    def __init__(self, d1=3, d2=2, family='linear',
                 constraint='orthogonal',**kwargs):
        """\
        Initializes Persp object, by setting the dimensions of the input and
        output spaces, the family of allowed perspective functions, and 
        constraints on the perspective parameters.
        
        Parameters:
        
        d1 : int
        Dimension of input space.
       
        d2 : int
        Dimension of output space.

        family : string
        Family of perspective functions. Currently, the only option is linear.

        constraint : None or string
        Constraint on the perspective parameters. Current options are None,
        'orthogonal', and 'similar'.
        """
        assert isinstance(d1,int) and d1>0; self.d1 = d1
        assert isinstance(d2,int) and d2>0; self.d2 = d2
        assert family in families; self.family = family
        assert constraint in constraint; self.constraint = constraint

        #setup family of projection functions:
        if self.family == 'linear':
            self.setup_linear()
        elif self.family == 'stereographic':
            self.shape = 1 #
        else:
            sys.exit('Persp family unknown.')

    def setup_linear(self):
        """\
        Sets up functions for linear projection family.
        """
        assert self.d1 >= self.d2
    
        self.shape = (self.d2,self.d1)
        self.p = lambda q,x : x @ q.T
        self.P = lambda q,X : X @ q.T
        self.dp = lambda q,x : q

        def special(number,method='identity'):
            if method == 'identity':
                q = np.identity(self.d1)
                q = q[0:self.d2]
                return [q]*number
            elif method == 'standard':
                Q = []
                for comb in itertools.combinations(range(self.d1),self.d2):
                    Q.append(np.identity(self.d1)[comb,:])
                assert len(Q) >= number
                return Q[0:number]
            elif method == 'cylinder':
                assert self.d1 == 3 and self.d2 == 2 ### generalize! ###
                Q = []
                for k in range(number):
                    theta = math.pi/number * k
                    Q.append(np.array([[math.cos(theta),math.sin(theta),0],
                                   [0,0,1]]))
                return Q
        self.special = special
            
        assert self.constraint in [None,'orthogonal','similar']
        if self.constraint is None:
            self.c = lambda x : x
            self.random = lambda : np.random.randn(self.d2,self.d1)
        elif self.constraint is 'orthogonal':
            def c(P):
                """\
                Returns nearest orthogonal matrix to P, that is, Q minimizing 
                |P-Q|_F such that Q @ Q.T = I and Q.T @ Q = I.
                """
                U,s,Vh = np.linalg.svd(P, full_matrices=False)
                return U @ Vh
            self.c = c
            self.random = lambda : scipy.stats.ortho_group.rvs(self.d1) \
                          [0:self.d2,:]
        elif self.constraint is 'similar':
            def c(P):
                """\
                Returns nearest scaled orthogonal matrix to P, that is, Q 
                minimizing |P-Q|_F such that Q @ Q.T = sI and Q.T @ Q = sI, for 
                some s >= 0.
                """
                U,s,Vh = np.linalg.svd(P, full_matrices=False)
                s = np.sum(s)/len(s)
                return s * U @ Vh
            self.c = c
            def random(rmax=2,rmin=0.5):
                q = scipy.stats.ortho_group.rvs(d1)[0:self.d2,:]
                q *= np.random.rand()*(rmax-rmin)+rmin
                return q
            self.random = random

    def check(self, q=None, Q=None, X=None):
        """\
        Checks that input perspective parameters or position array have the 
        correct form.

        Parameters

        q : numpy array
        Array with parameters for one projection.

        Q : list
        List with parameter arrays for multiple projections.

        X : ndarray
        Input coordinate or array of input coordinates.
        """
        if q is not None:
            assert isinstance(q,np.ndarray)
            assert q.shape == self.shape
        if Q is not None:
            assert isinstance(Q,list) or isinstance(Q,np.ndarray)
            for q in Q:
                assert isinstance(q,np.ndarray)
                assert q.shape == self.shape
        if X is not None:
            assert isinstance(X,np.ndarray)
            if X.ndim == 1:
                assert len(X) == self.d1
            else:
                assert X.ndim == 2
                assert X.shape[1] == self.d1

    def project(self, q, X):
        """\
        Returns projected data.
        
        Parameters:
        
        q : array or list
        Projection parameters or list of projection parameters.

        X : array (N x d1)
        Coordinate array.

        Returns:
        
        Y : array or list
        Projected coordinates p_q(X) or [p_q1(X),...,p_qK(X)].
        """
        if isinstance(q,np.ndarray) and q.shape==self.shape:
            Y = self.P(q,X)
        else:
            Y = []
            for qk in q:
                Y.append(self.P(qk,X))
            if isinstance(q,np.ndarray):
                Y = np.array(Y)
        return Y
            
    def gradient(self, q, x):
        if isinstance(q,np.ndarray) and q.shape==self.shape:
            dpx = self.dp(q,x)
        else:
            dpx = []
            for qk in q:
                dpx.append(self.dp(q,x))
            if isinstance(q,np.ndarray):
                dpx = np.array(dpx)
        return dpx

    def restrict(self, q):
        if isinstance(q,np.ndarray) and q.shape==self.shape:
            qq = self.c(q)
        else:
            qq = []
            for qk in q:
                qq.append(self.c(qk))
            qq = np.array(qq)
        return qq

    def generate(self, number=3, method='random',**kwargs):
        """\
        Returns a list of projection parameter arrays.

        Parameters:

        number : int
        Number of parameter arrays.

        method : string
        Method used to generate parameters. Choices that work for every set of
        projection/perspective families are ['identity','random']. Other choices
        are dependent on the family and constraints.
        """
        if method is None:
            method = 'random'
        if method == 'random':
            Q = []
            for i in range(number):
                Q.append(self.random())
        else:
            Q = self.special(number,method=method)
        return Q

### Functions returning special parameters ###.

def special_parameters(d1,d2,number,method='identity'):
    if method == 'identity':
        assert d1 == d2
        Q = [np.identity(d1)]*number
    elif method == 'same':
        Q = [np.identity(d1)[0:d2,:]]*number
    elif method == 'standard':
        assert d1 >= d2
        Q = []
        for comb in itertools.combinations(range(d1),d2):
            Q.append(np.identity(d1)[comb,:])
        assert len(Q) >= number
        Q = Q[0:number]
    elif method == 'cylinder':
        assert d1 == 3 and d2 == 2 ### to be generalized ###
        Q = []
        for k in range(number):
            theta = math.pi/number * k
            Q.append(np.array([[math.cos(theta),math.sin(theta),0],
                               [0,0,1]]))
    else:
        sys.exit('perspective.special_parameters() method does not exist.')
    return Q

def random_parameters(d1,d2,number,method='orthogonal'):
    if method == 'uniform':
        Q = []
        for k in range(number):
            Q.append(np.random.rand(d2,d1)-0.5)
    elif method == 'normal':
        Q = []
        for k in range(number):
            Q.append(np.random.randn(d2,d1))
    elif method == 'orthogonal':
        Q = []
        for k in range(number):
            Q.append(scipy.stats.ortho_group.rvs(d1)[0:d2,:])
    else:
        sys.exit('perspective.random_parameters() method does not exist.')
    return Q
