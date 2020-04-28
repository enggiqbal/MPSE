### MDS implementation ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import misc, multigraph, gd, plots

def stress_function(X,D,normalize=True,estimate=True,**kwargs):
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
                import scipy.spatial
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
                for i in range(D['edge_number']):
                    i1,i2 = D['edge_list'][i]
                    dXij = np.linalg.norm(X[i1]-X[i2])
                    stress += (D['dissimilarity_list'][i]-dXij)**2
                stress = math.sqrt(stress/D['edge-number'])/D['rms']
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
                for i in range(edge_number):
                    i1,i2 = D['edge_list'][inds[i]]
                    dX = np.linalg.norm(X[i1]-X[i2])
                    stress += (dX-D['dissimilarity_list'][i])**2
            stress = math.sqrt(stress/edge_number)/D['rms']
    return stress

def F(X,D,normalize=True):
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
        for i in range(D['edge_number']):
            i1,i2 = D['edge_list'][i]
            Xij = X[i1]-X[i2]
            dij = np.linalg.norm(Xij)
            diffij = dij-D['dissimilarity_list'][i]
            fX += diffij**2
            dXij = 2*diffij/dij*Xij
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
    def __init__(self, D, dim=2, verbose=0, title='',level=0,**kwargs):
        """\
        Initializes MDS object.

        Parameters:

        D : dictionary or numpy array
        Either i) a dictionary with the lists of edges, distances, and weights 
        as described in dissimilarities.py or ii) a dissimilarity
        matrix.

        dim : int > 0
        Embedding dimension.

        verbose : int >= 0
        Print status of methods in MDS object if verbose > 0.

        title : string
        Title assigned to MDS object.
        """
        if verbose > 0:
            print('  '*level+'mds.MDS('+title+'):')
        self.verbose = verbose; self.title = title; self.level = level

        self.D = multigraph.attribute_setup(D,**kwargs)
        self.N = self.D['node_number']; self.NN = self.D['edge_number']
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim
        
        self.f = lambda X, D=self.D, **kwargs : stress_function(X,D,**kwargs)
        self.F = lambda X, D=self.D, **kwargs : F(X,D,**kwargs)

        self.initial_cost = None
        self.H = {}
        
        if verbose > 0:
            print('  '*level+'  dissimilarity stats:')
            print('  '*level+f'    number of points : {self.N}')
            print('  '*level+f'    number of edges : {self.NN}')
            print('  '*level+f'    dissimilarity rms : {self.D["rms"]:0.2e}')
            print('  '*level+'  embedding stats:')
            print('  '*level+f'    dimension : {self.dim}')
            
    def initialize(self, X0=None, title='',**kwargs):
        """\
        Set initial embedding.

        Parameters:

        X0 : numpy array or None
        Initial embedding. If set to None, the initial embedding is produced 
        randomly using misc.initial_embedding().
        """
        if self.verbose > 0:
            print(f'  MDS.initialize({self.title} - {title}):')
            
        if X0 is None:
            X0 = misc.initial_embedding(self.N,dim=self.dim,
                                        radius=self.D['rms'],**kwargs)
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
            print(f'    initial stress : {self.cost:0.2e}')

    def update(self,H=None,**kwargs):
        self.cost = self.f(self.X,**kwargs)
        if self.initial_cost is None:
            self.initial_cost = self.cost
        if H is not None:
            if bool(self.H) is True:
                H['costs'] = np.concatenate((self.H['costs'],H['costs']))
                H['steps'] = np.concatenate((self.H['steps'],H['steps']))
                H['lrs'] = np.concatenate((self.H['lrs'],H['lrs']))
                H['grads'] = np.concatenate((self.H['grads'],H['grads']))
                H['iterations'] = self.H['iterations']+H['iterations']
            self.H = H        

    def forget(self):
        self.X = self.X0; self.H = {}
        self.update()

    ### Methods to update MDS embedding ###
    
    def subsample_generator(self,edge_proportion=None,average_neighbors=None,
                            **kwargs):
        if edge_proportion is None and average_neighbors is None:
            return None
        else:
            Xi = lambda : multigraph.\
                attribute_sample(self.D,edge_proportion=edge_proportion,
                                 average_neighbors=average_neighbors,**kwargs)
            return Xi

    def gd(self, scheme='mm',**kwargs):
        if hasattr(self,'X') is False:
            self.initialize(title='automatic',**kwargs)
        if self.verbose > 0:
            print(f'  MDS.gd({self.title}):')
            #print(f'    step rule : {step_rule}')
            #print(f'    edge probability : {edge_probability}')
            print(f'    initial stress : {self.cost:0.2e}')

        Xi = self.subsample_generator(**kwargs)
        F = lambda X, xi=self.D : self.F(X,D=xi)
        self.X, H = gd.single(self.X,F,Xi=Xi,scheme=scheme,**kwargs)
        self.update(H=H,**kwargs)
        if self.verbose > 0:
            print(f'    final stress : {self.cost:0.2e}')

    ### Plotting methods ###

    def figureX(self,title='',edges=False,node_color=None,axis=True,plot=True,
                ax=None):
        assert self.dim >= 2
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        if edges is True:
            edges = self.D['edge_list']
        elif edges is False:
            edges = None
        if node_color is None:
            node_color = self.D['node_colors']
        plots.plot2D(self.X,edges=edges,colors=node_color,axis=axis,ax=ax,
                     title=title)
        if plot is True:
            plt.draw()
            plt.pause(1)

    def figureH(self,title='computations',plot=True,ax=None):
        assert hasattr(self,'H')
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        ax.semilogy(self.H['costs'],label='stress',linewidth=3)
        ax.semilogy(self.H['grads'],label='grad size')
        ax.semilogy(self.H['lrs'],label='lr')
        ax.semilogy(self.H['steps'],label='step size')
        ax.legend()
        ax.set_title(title)
        if plot is True:
            plt.draw()
            plt.pause(1.0)
                                   
### TESTS ###

def disk(N=128,**kwargs):
    print('\n***mds.disk()***')
    
    X = misc.disk(N,2); colors = misc.labels(X)
    diss = multigraph.DISS(N)
    diss.add_feature(X,node_colors=colors)
    D = diss.return_attribute()

    title = 'basic disk example'
    mds = MDS(D,dim=2,verbose=1,title=title)
    mds.initialize(**kwargs)

    fig, ax = plt.subplots(1,3,figsize=(9,3))
    fig.suptitle('MDS - disk data')
    fig.subplots_adjust(top=0.80)
    mds.figureX(title='initial embedding',ax=ax[0])
    mds.gd(verbose=2,max_iter=30,**kwargs)
    mds.gd(verbose=2,max_iter=30,average_neighbors=6)
    mds.gd(verbose=2,average_neighbors=None)
    mds.figureH(ax=ax[1])
    mds.figureX(title='final embedding',ax=ax[2])
    plt.draw()
    plt.pause(1.0)

if __name__=='__main__':

    print('mview.mds : running tests')
    disk(N=100,scheme='mm',average_neighbors=1)
    plt.show()
