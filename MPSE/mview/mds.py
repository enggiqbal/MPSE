### MDS implementation ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import misc, multigraph, gd, plots

def f(X,D,estimate=True):
    """\

    Normalized MDS stress function.

    Parameters:

    X : numpy array
    Position/coordinate/embedding array for nodes 0,1,...,N-1

    D : dictionary
    Dissimilarity graph, containing edges, distances, and weights.

    Returns:

    stress : float
    MDS stress at X.
    """
    if estimate is True and D['edges']>64*63/2:
        stress = 0
        if D['complete'] is True:
            edges = misc.random_triangular(D['nodes'],int(64*63/2))
            for i1,i2 in edges:
                dX = np.linalg.norm(X[i1]-X[i2])
                stress += (dX-D['dfunction'](i1,i2))**2
        else:
            inds = np.random.choice(D['edges'],int(64*63/2))
            for i in range(int(64*63/2)):
                i1,i2 = inds[i]
                dX = np.linalg.norm(X[i1]-X[i2])
                stress += (dX-D['dlist'][i])**2
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
                for i in range(D['nodes']):
                    for j in range(D['nodes']):
                        dXij = np.linalg.norm(X[i]-X[j])
                        stress += (D['dfunction'](i,j)-dXij)**2
                stress = math.sqrt(stress) / D['normalization']
        else:
            if D['weighted'] is False:
                stress = 0
                for (i,j) in D['edges']:
                    dXij = np.linalg.nodm(X[i]-X[j])
                    stress += (D['dissimilarities'][i,j]-dXij)
                stress = math.sqrt(stress)/D['normalization']
    return stress

def fF(X,D):
    """\
    Returns exact stress and gradient for embedding X with target distances D.
    """
    if D['normalization'] is None:
        D['normalization'] = f(X,D)
    
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
    return fX, dfX

def sF(X,D,number_of_edges,replace=False):
    """\
    Returns stress and gradient for embedding X.
    """
    N, dim = X.shape

    #Create list of edges to be used in computation:
    if D['complete'] is True:
        NN = round(number_of_edges)
        edges = misc.random_triangular(N,NN,replace=replace)
        dlist = np.empty(NN)
        for i in range(NN):
            edge = edges[i]
            dlist[i] = D['dfunction'](int(edge[0]),int(edge[1]))
    else:
        NN = int(number_of_edges)
        inds = np.random.choice(len(D['edges']),NN)
        edges = D['edges'][inds]
        dlist = D['dlist'][inds]
    normalization = np.linalg.norm(dlist)

    fX = 0
    dfX = np.zeros(X.shape)
    for i in range(NN):
        edge = edges[i]
        Xij = X[edge[0]]-X[edge[1]]
        d = np.linalg.norm(Xij)
        diff = d-dlist[i]
        fX += diff**2
        dX = (2*diff/d)*Xij
        dfX[edge[0]] += dX
        dfX[edge[1]] -= dX
    fX /= normalization
    dfX /= normalization

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

def graph_f(X,D):
    """\

    Normalized MDS stress function.

    Parameters:

    X : numpy array
    Position/coordinate/embedding array for nodes 0,1,...,N-1

    D : dictionary
    Dissimilarity graph, containing edges, distances, and weights.

    Returns:

    stress : float
    MDS stress at X.
    """
    e = D['edges']; d = D['distances']; w = D['weights']
    stress = 0
    for (i,j), Dij, wij in zip(e,d,w):
        dij = np.linalg.norm(X[i]-X[j])
        stress += wij*(Dij-dij)**2
    stress /= D['normalization']
    return math.sqrt(stress)

def graph_F(X,D,edge_probability=None):
    """\
    Returns (normalized) MDS stress and gradient.
    
    Parameters:

    X : numpy array
    Positions/embedding array.

    D : dictionary
    Dictionary containing list of dissimilarities.

    Returns:

    fX : float
    MDS stress at X.
    
    dfX : numpy array
    MDS gradient at X.
    """
    e = D['edges']; d = D['distances']; w = D['weights']
    fX = 0; dfX = np.zeros(X.shape)
    if edge_probability is None:
        for n in range(len(e)):
            i,j = e[n]; Dij = d[n]; wij = w[n]
            Xij = X[i]-X[j]
            dij = np.linalg.norm(Xij)
            diffij = dij-Dij
            fX += wij*diffij**2
            dXij = wij*2*diffij/dij*Xij
            dfX[i] += dXij
            dfX[j] -= dXij
        fX /= D['normalization']
        dfX /= D['normalization']
    else:
        assert isinstance(edge_probability,numbers.Number)
        assert 0<edge_probability<=1
        normalization = 0
        for n in range(len(e)):
            if np.random.rand() <= edge_probability:
                i,j = e[n]; Dij = d[n]; wij = w[n]
                Xij = X[i]-X[j]
                dij = np.linalg.norm(Xij)
                diffij = dij-Dij
                fX += wij*diffij**2
                dXij = wij*2*diffij/dij*Xij
                dfX[i] += dXij
                dfX[j] -= dXij
                normalization += wij*Dij**2
        fX /= normalization
        dfX /= normalization
    return math.sqrt(fX), dfX

def F(X,D,stochastic=None,**kwargs):
    if stochastic is None:
        fX, dfX = fF(X,D)
    else:
        if stochastic >= 1:
            number_of_edges = round(len(X)*stochastic/2)
        elif 0 < stochastic < 1:
            number_of_edges = round(D['edges']*stochastic)
        fX, dfX = sF(X,D,number_of_edges)
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
        self.D['normalization'] = self.D['rms']*self.D['edges']
        self.N = D['nodes']; self.NN = D['edges']
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim
        
        self.f = lambda X: f(X,self.D,**kwargs)
        self.F = lambda X, **kwargs : F(X,self.D,**kwargs)

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

    def gd(self, step_rule='mm', min_step=1e-4,
           title='', **kwargs):
        if hasattr(self,'X') is False:
            self.initialize(title='automatic')
        if self.verbose > 0:
            print(f'  MDS.gd({self.title} - {title}):')
            print(f'    step rule : {step_rule}')
            #print(f'    edge probability : {edge_probability}')
            print(f'    initial stress : {self.cost:0.2e}')
        F = lambda X: self.F(X,**kwargs)
        self.X, H = gd.single(self.X,F,step_rule=step_rule,
                              min_step=min_step,**kwargs)
        self.update(H=H)
        if self.verbose > 0:
            print(f'    final stress : {self.cost:0.2e}')

    ### Plotting methods ###

    def figureX(self,title='',edges=False,colors=False,axis=True,plot=True,
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
        if colors is True:
            colors = self.D['ncolor']
        elif colors is False:
            colors = None
        plots.plot2D(self.X,edges=edges,colors=colors,axis=axis,ax=ax,
                     title=title)
        if plot is True:
            plt.draw()
            plt.pause(1)

    def figureH(self,title='computations',plot=True,ax=None):
        assert hasattr(self,'H')
        if ax is None:
            fig, ax = plt.subplots()
        plots.plot_cost(self.H['costs'],self.H['steps'],title=title,ax=ax)
        if plot is True:
            plt.draw()
            plt.pause(1.0)

    def figure(self,title='mds computation & embedding',labels=None,
               plot=True):
        assert self.dim >= 2
        fig,axs = plt.subplots(1,2)
        plt.suptitle(title+f' - stress = {self.cost:0.2e}')
        self.figureH(ax=axs[0])
        self.figureX(edges=True,ax=axs[1])
        if plot is True:
            plt.draw()
            plt.pause(1.0)
        return fig
                                   
### TESTS ###

def disk(N=100,**kwargs):
    print('\n***mds.disk()***')
    
    X = misc.disk(N,2); colors = misc.labels(X)
    DD = multigraph.DISS(N,ncolor=colors)
    DD.from_features(X)
    D = DD.return_attribute()
    
    title = 'basic disk example'
    mds = MDS(D,dim=2,verbose=1,title=title)
    mds.initialize(X0=X)
    mds.figureX(title='original data',colors=True)
    mds.initialize()
    mds.figureX(title='initial embedding',colors=True)
    mds.gd(max_iter=500,min_cost=1e-3,min_step=1e-5,verbose=2,plot=True,
           **kwargs)
    mds.figureX(title='final embedding',colors=True)
    plt.show()

def example_disk(N=100,dim=2,**kwargs):
    print('\n***mds.example_disk()***')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y)
    
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1],c=colors)
    plt.title('original data')
    plt.draw()
    plt.pause(1)
    
    D = multigraph.graph_from_coordinates(Y,colors=colors)
    title = 'basic disk example'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    mds.figureX(title='initial embedding',colors=True)
    mds.gd(edge_probability=None,verbose=2,plot=True,**kwargs)
    mds.figureX(title='final embedding',colors=True)
    mds.figureH()
    plt.show()

if __name__=='__main__':

    disk(N=1000,stochastic=128)
