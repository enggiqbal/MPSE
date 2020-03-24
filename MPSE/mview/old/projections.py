import os, sys
import numpy as np
import scipy.stats
import math, numbers

class Proj(object):
    """\
    Class specifying the set of allowed projection functions and containing
    methods to call and update a list of projection parameters.
    """

    def __init__(self, dimX=3, dimY=2, family='linear',
                 restriction='orthogonal', set_params=True, **kwargs):
        """\
        Initializes Proj object, by setting the dimensions of the input and
        output spaces, the family of allowed projection functions, and 
        restrictions on the projection parameters.

        Parameters:
        
        dimX : int
        Dimension of input space.
       
        dimY : int
        Dimension of output space.

        family : string
        Family of projection functions. Current options is linear'.

        restriction : None or string
        Restriction on the projection parameters. Current options are None,
        'orthogonal', and 'similar'.
        """
        self.dimX = dimX
        self.dimY = dimY
        self.family = family
        self.restriction = restriction

        assert family in ['linear']
        self.setup_projection_functions()
                
        if restriction is None:
            self.rfunction = lambda x: x
        elif restriction is 'orthogonal':
            self.rfunction = restriction_orthogonal
        elif restrict is 'similar':
            self.rfunction = restriction_similar
        else:
            sys.exit('Restriction unknown')

        self.generate_params_list(**kwargs)

    def setup_projection_functions(self):

        if self.family == 'linear':
            self.param_shape = (self.dimY,self.dimX)
            self.projection_function = lambda q,x: x @ q.T
            self.array_projection_function = lambda q,X : X @ q.T
            self.gradient_function = lambda q,x: q
        else:
            sys.exit('Family of projections unknown.')

    def check(self, param=None, params_list=None, X=None, verbose=0):
        """\
        Checks that input projection parameters or position array have the 
        correct form.

        --- Parameters ---

        param : numpy array
        Array with parameters for one projection.

        param_list : list
        List with parameter arrays for multiple projections.

        X : ndarray
        Input coordinate or array of input coordinates.
        """
        if param is not None:
            assert isinstance(param,np.ndarray)
            assert param.shape == self.param_shape
        if params_list is not None:
            assert isinstance(params_list,list) or \
                isinstance(params_list,np.ndarray)
            for param in params_list:
                assert isinstance(param,np.ndarray)
                assert param.shape == self.param_shape
        if X is not None:
            assert isinstance(X,np.ndarray)
            if X.ndim == 1:
                assert len(X) == self.dimX
            else:
                assert X.ndim == 2
                assert X.shape[1] == self.dimX
        if verbose > 0:
            print('Projs.check(): complete')

    def generate_params_list(self, number=3, special=None, random='orthogonal'):
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
            if special == 'identity':
                assert self.dimX == self.dimY
                params = [np.identity(self.dimX)]*number
            elif special == 'standard':
                params = standard(self.dimX,self.dimY,number)
            elif special == 'cylinder':
                params = cylinder(self.dimX,self.dimY,number)
            else:
                sys.exit('Special parameter choice does not exist.')
        elif random is not None:
            params = random_parameters(self.dimX,self.dimY,number,method=random)
        else:
            sys.exit('No parameter family was specified.')
        return params

    def set_params_list(self, params_list=None, number=3, special=None,
                       random='uniform'):
        """\
        Saves list with parameters array into Proj object. For use when the
        list of projection parameters is fixed.
        """
        if params_list is not None:
            self.check(params_list=params_list)
        else:
            params_list = self.generate_params_list(number,special,random)
        self.params_list = params_list
        self.params_number = number
            
    def project(self, X, param=None, params_list=None):
        """\
        Returns projected data.
        """
        if param is not None:
            Y = self.projection_function(param,X)
        elif params_list is not None:
            Y = []
            for param in params_list:
                Y.append(self.array_projection_function(param,X))
        elif hasattr(self,'params_list'):
            Y = []
            for param in self.params_list:
                Y.append(self.array_projection_function(param,X))
        else:
            sys.exit('Error: Proj.project() - shape of Q is incorrect.')
        return Y

    def compute_gradient(self, x, param=None, params_list=None):
        if param is not None:
            J = self.gradient_function(param,x)
        elif params_list is not None:
            J = []
            for param in params_list:
                J.append(self.gradient_function(param,x))
        return J

    def restrict(self, param=None, param_list=None):
        if Q.shape==self.param_shape:
            Qr = self.restriction(Q)
        else:
            Qr = np.empty(Q.shape)
            for i in range(self.number):
                Qr[i] = self.restriction(Q[i])
        return Qr

    def plot3D(self, X, Q=None, Y=None, labels=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        if Q is None:
            Q = self.params_list
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

    
### Restriction mappings ###

# Here, the parameter P is 'projected' to the subset of parameters that is
# specified. Hence, the parameter choice Q minimizing |P-Q|_F is returned.

def restriction_orthogonal(P):
    """\
    Returns nearest orthogonal matrix to P, that is, Q minimizing |P-Q|_F such
    that Q @ Q.T = I and Q.T @ Q = I.
    """
    U,s,Vh = np.linalg.svd(P, full_matrices=False)
    return U @ Vh

def restriction_similar(P):
    """\
    Returns nearest scaled orthogonal matrix to P, that is, Q minimizing |P-Q|_F
    such that Q @ Q.T = sI and Q.T @ Q = sI, for some s >= 0.
    """
    U,s,Vh = np.linalg.svd(P, full_matrices=False)
    s = np.sum(s)/len(s)
    return s * U @ Vh

### Functions returning special parameters ###

# These are functions returning special choices for the projections' parameters.

def standard(dimX,dimY,K):
    import itertools
    assert dimX >= dimY
    par = []
    for comb in itertools.combinations(range(dimX),dimY):
        par.append(np.identity(dimX)[comb,:])
    assert len(par) >= K
    return par[0:K]

def cylinder(dimX,dimY,K): ### TBD ###
    assert dimX == 3 and dimY == 2 ### to be generalized ###
    params_list = []
    for k in range(K):
        theta = math.pi/K * k
        params_list.append(np.array([[math.cos(theta),math.sin(theta),0],
                                     [0,0,1]]))
    return params_list

### Generation of parameters list using random methods ###

def random_uniform(dimX,dimY):
    return np.random.rand(dimY,dimX)-0.5

def random_normal(dimX,dimY):
    return np.random.randn(dimY,dimX)

def random_orthogonal(dimX,dimY):
    return scipy.stats.ortho_group.rvs(dimX)[0:dimY,:]

random_methods = {
    'uniform' : random_uniform,
    'normal' : random_normal,
    'orthogonal' : random_orthogonal
    }

def random_parameters(dimX,dimY,K,method='orthogonal'):
    method = random_methods[method]
    params_list = []
    for k in range(K):
        params_list.append(method(dimX,dimY))
    return params_list
