import os, sys
import numpy as np
import scipy.stats
import math, numbers, itertools

families = ['linear']
constraints = [None,'orthogonal','similar']

class Persp(object):
    """\
    Class use to define the desired collection of perspective functions and    
    containing methods to call perspective functions in such collection.
    """
    
    def __init__(self, dimX=3, dimY=2, family='linear',
                 constraint='orthogonal'):
        """\
        Initializes Persp object, by setting the dimensions of the input and
        output spaces, the family of allowed perspective functions, and 
        constraints on the perspective parameters.
        
        Parameters:
        
        dimX : int
        Dimension of input space.
       
        dimY : int
        Dimension of output space.

        family : string
        Family of perspective functions. Currently, the only option is linear.

        constraint : None or string
        Constraint on the perspective parameters. Current options are None,
        'orthogonal', and 'similar'.
        """
        assert isinstance(dimX,int) and dimX>0; self.dimX = dimX
        assert isinstance(dimY,int) and dimY>0; self.dimY = dimY
        assert family in families; self.family = family
        assert constraint in constraint; self.constraint = constraint
        
        self.setup_perspective_functions()
        self.setup_constraint_functions()

        self.fixed_Q = False
        
    def setup_perspective_functions(self):
        if self.family == 'linear':
            self.shape = (self.dimY,self.dimX)
            self.p = lambda q,x: x @ q.T
            self.P = lambda q,X : X @ q.T
            self.dp = lambda q,x: q
        else:
            sys.exit('Persp family unknown.')

    def setup_constraint_functions(self):
        if self.constraint is None:
            self.c = lambda x: x
        elif self.constraint is 'orthogonal':
            def orthogonal(P):
                """\
                Returns nearest orthogonal matrix to P, that is, Q minimizing 
                |P-Q|_F such that Q @ Q.T = I and Q.T @ Q = I.
                """
                U,s,Vh = np.linalg.svd(P, full_matrices=False)
                return U @ Vh
            self.c = orthogonal
        elif self.constraint is 'similar':
            def similar(P):
                """\
                Returns nearest scaled orthogonal matrix to P, that is, Q 
                minimizing |P-Q|_F such that Q @ Q.T = sI and Q.T @ Q = sI, for 
                some s >= 0.
                """
                U,s,Vh = np.linalg.svd(P, full_matrices=False)
                s = np.sum(s)/len(s)
                return s * U @ Vh
            self.c = similar
        else:
            sys.exit('Persp constraint unknown')

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
                assert len(X) == self.dimX
            else:
                assert X.ndim == 2
                assert X.shape[1] == self.dimX

    def generate_Q(self, number=3, special=None, random='orthogonal'):
        """\
        Returns a parameter list.

        Parameters:
        
        number : int
        Number of parameter arrays

        special : string
        Returns a special list of parameters. Current options are 'standard' and
        'cylinder'.

        random : string
        If special is not specified, returns a random list of parameters, as
        specified by random. Current options are 'uniform'.
        """
        if special is not None:
            Q = special_parameters(self.dimX,self.dimY,number,method=special)   
        elif random is not None:
            Q = random_parameters(self.dimX,self.dimY,number,method=random)
        else:
            sys.exit('No specifications for Q')
        return Q

    def fix_Q(self, Q=None, **kwargs):
        """\
        Saves list with parameters array into Proj object. For use when the
        list of projection parameters is fixed.
        """
        if Q is None:
            Q = self.generate_Q(**kwargs)
        else:
            self.check(Q=Q)
     
        self.Q = Q
        self.K = len(Q)
        self.fixed_Q = True
            
    def compute_Y(self, X, q=None, Q=None):
        """\
        Returns projected data.
        """
        if q is not None:
            Y = self.P(q,X)
        elif Q is not None:
            Y = []
            for q in Q:
                Y.append(self.P(q,X))
        else:
            assert self.fixed_Q is True
            Y = []
            for q in self.Q:
                Y.append(self.P(q,X))
        return Y

    def compute_gradient(self, x, param=None, Q=None):
        if param is not None:
            J = self.dp(param,x)
        elif Q is not None:
            J = []
            for param in Q:
                J.append(self.dp(param,x))
        return J

    ### Unnecessary ?
    def restrict(self, param=None, param_list=None):
        if Q.shape==self.shape:
            Qr = self.constraint(Q)
        else:
            Qr = np.empty(Q.shape)
            for i in range(self.number):
                Qr[i] = self.constraint(Q[i])
        return Qr

    def plot3D(self, X, Q=None, Y=None, labels=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        if Q is None:
            Q = self.Q
        fig, axs = plt.subplots(1,self.number,sharex=True)
        plt.tight_layout()
        if Y is None:
            Y = self.project(Q,X)
        if labels is None:
            labels = range(self.number)
        for i in range(self.number):
            axs[i].scatter(Y[i][:,0],Y[i][:,1])
            axs[i].set_aspect(1.0)
            axs[i].set_title(f'Projection {labels[i]}')
        plt.suptitle('Projected Data')
        plt.show(block=False)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:,0],X[:,1],X[:,2])
        ax.set_aspect(1.0)
        plt.show()


### Functions returning special parameters ###.

def special_parameters(dimX,dimY,number,method='identity'):
    if method == 'identity':
        assert dimX == dimY
        Q = [np.identity(dimX)]*number
    elif method == 'same':
        Q = [np.identity(dimX)[0:dimY,:]]*number
    elif method == 'standard':
        assert dimX >= dimY
        Q = []
        for comb in itertools.combinations(range(dimX),dimY):
            Q.append(np.identity(dimX)[comb,:])
        assert len(Q) >= number
        Q = Q[0:number]
    elif method == 'cylinder':
        assert dimX == 3 and dimY == 2 ### to be generalized ###
        Q = []
        for k in range(number):
            theta = math.pi/number * k
            Q.append(np.array([[math.cos(theta),math.sin(theta),0],
                               [0,0,1]]))
    else:
        sys.exit('perspective.special_parameters() method does not exist.')
    return Q

def random_parameters(dimX,dimY,number,method='orthogonal'):
    if method == 'uniform':
        Q = []
        for k in range(number):
            Q.append(np.random.rand(dimY,dimX)-0.5)
    elif method == 'normal':
        Q = []
        for k in range(number):
            Q.append(np.random.randn(dimY,dimX))
    elif method == 'orthogonal':
        Q = []
        for k in range(number):
            Q.append(scipy.stats.ortho_group.rvs(dimX)[0:dimY,:])
    else:
        sys.exit('perspective.random_parameters() method does not exist.')
    return Q
