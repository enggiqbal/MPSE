### MDS implementation ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import misc, multigraph, gd, plots

def stress_function(X,D,estimate=True):
    """\

    Normalized MDS stress function.

    Parameters:

    X : numpy array
    Position/coordinate/embedding array.

    D : numpy array or dictionary
    Either a dissimilarity matrix or a dissimilarity dictionary as specified
    in mview.multigraph.

    estimate : boolean
    If set to True, it estimates stress to reduce computation time.

    Returns:

    stress : float
    MDS stress at X.
    """
    if isinstance(D,np.ndarray):
        if estimate is False:
            dX = sp.spatial.distance_matrix(X,X)
            stress = np.linalg.norm(D-dX)
            stress /= np.linalg.norm(D)
        return stress
            
    if estimate is True and D['edge_number']>64*63/2:
        stress = 0
        if D['complete'] is True:
            edges = misc.random_triangular(D['node_number'],int(64*63/2))
            for i1,i2 in edges:
                dX = np.linalg.norm(X[i1]-X[i2])
                stress += (dX-D['dfunction'](i1,i2))**2
        else:
            inds = np.random.choice(D['edge_number'],int(64*63/2),replace=False)
            for i in range(int(64*63/2)):
                i1,i2 = D['edge_list'][inds[i]]
                dX = np.linalg.norm(X[i1]-X[i2])
                stress += (dX-D['dissimilarity_list'][i])**2
        stress = math.sqrt(stress/(64*63*2))
    else:
        if D['type'] == 'matrix':
            if D['weighted'] is False:
                import scipy.spatial
                dX = scipy.spatial.distance_matrix(X,X)
                stress = np.linalg.norm(D['matrix']-dX)
                stress /= D['normalization']
        elif D['complete'] is True:
            if D['weighted'] is False:
                stress = 0
                for i in range(D['node_number']):
                    for j in range(D['node_number']):
                        dXij = np.linalg.norm(X[i]-X[j])
                        stress += (D['dfunction'](i,j)-dXij)**2
                stress = math.sqrt(stress) / D['normalization']
        else:
            if D['weighted'] is False:
                stress = 0
                for i in range(D['edge_number']):
                    i1,i2 = D['edge_list'][i]
                    dXij = np.linalg.norm(X[i1]-X[i2])
                    stress += (D['dissimilarity_list'][i]-dXij)**2
                stress = math.sqrt(stress)/np.linalg.norm(D['dissimilarity_list'])
    return stress

def F(X,D):
    """\
    Returns exact stress and gradient for embedding X with target distances D.
    """
    #if D['normalization'] is None:
    #    D['normalization'] = f(np.zeros(X.shape),D)
    
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

        fX = math.sqrt(fX) / D['normalization']
        dfX /= D['normalization']
    else:
        assert D['type'] == 'graph'
        dfX = np.zeros(X.shape)
        f0 = np.linalg.norm(D['dissimilarity_list'])
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
        fX = math.sqrt(fX)/f0
        dfX /= f0
        
    return fX, dfX

def old():

    #Compute differences in positions for node pairs of edges:
    Xij = np.empty((NN,dim))
    for i in range(NN):
        edge = edges[i]
        Xij[i] = X[edge[0]]-X[edge[1]]
        
    dfX = np.zeros(X.shape)
    d = np.linalg.norm(Xij,axis=1)
    diff = d-dlist
    fX = np.linalg.norm(diff)
    dX = (2*diff/d)[:, np.newaxis]*Xij
    
    for i in range(NN):
        edge = edges[i]
        dfX[edge[0]] += dX[i]
        dfX[edge[1]] -= dX[i]
    fX /= normalization
    dfX /= normalization

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
        self.D['rms'] = multigraph.attribute_rms(self.D,**kwargs)
        self.D['normalization'] = self.D['rms']*self.D['edge_number']
        self.N = self.D['node_number']; self.NN = self.D['edge_number']
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim
        
        self.f = lambda X, D=self.D, **kwargs : stress_function(X,D,**kwargs)
        self.F = lambda X, D=self.D, **kwargs : F(X,D,**kwargs)

        self.H = {}
        
        if verbose > 0:
            print('  '*level+'  dissimilarity stats:')
            print('  '*level+f'    number of points : {self.N}')
            print('  '*level+f'    number of edges : {self.NN}')
            print('  '*level+f'    dissimilarity rms : {self.D["rms"]:0.2e}')
            print('  '*level+f'    normalization factor : {self.D["normalization"]:0.2e}')
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
        self.update()
        
        self.X0 = self.X.copy()
        
        if self.verbose > 0:
            print(f'    initial stress : {self.cost:0.2e}')

    def update(self,H=None):
        self.cost = self.f(self.X)
        if H is not None:
            if bool(self.H) is True:
                H['costs'] = np.concatenate((self.H['costs'],H['costs']))
                H['steps'] = np.concatenate((self.H['steps'],H['steps']))
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
            self.initialize(title='automatic')
        if self.verbose > 0:
            print(f'  MDS.gd({self.title}):')
            #print(f'    step rule : {step_rule}')
            #print(f'    edge probability : {edge_probability}')
            print(f'    initial stress : {self.cost:0.2e}')

        Xi = self.subsample_generator(**kwargs)
        F = lambda X, xi=self.D : self.F(X,D=xi)
        self.X, H = gd.single(self.X,F,Xi=Xi,scheme=scheme,**kwargs)
        self.update(H=H)
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
            edges = self.D['edges']
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

def disk(N=100,**kwargs):
    print('\n***mds.disk()***')
    
    X = misc.disk(N,2); colors = misc.labels(X)
    diss = multigraph.DISS(N)
    diss.add_feature(X,node_colors=colors)
    D = diss.return_attribute()
    
    title = 'basic disk example'
    mds = MDS(D,dim=2,verbose=1,title=title)
    mds.initialize()
    fig, ax = plt.subplots(1,3,figsize=(9,3))
    fig.suptitle('MDS - disk data')
    fig.subplots_adjust(top=0.80)
    mds.figureX(title='initial embedding',ax=ax[0])
    mds.gd(max_iter=100,verbose=2,**kwargs)
    mds.figureH(ax=ax[1])
    mds.figureX(title='final embedding',ax=ax[2])
    plt.show()

if __name__=='__main__':

    print('mview.mds : running tests')
    disk(N=100,scheme='mm',average_neighbors=2,lr=1)
