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
            inds = np.random.choice(D['edges'],int(64*63/2),replace=False)
            for i in range(int(64*63/2)):
                i1,i2 = D['elist'][inds[i]]
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
                for i in range(D['edges']):
                    i1,i2 = D['elist'][i]
                    dXij = np.linalg.norm(X[i1]-X[i2])
                    stress += (D['dlist'][i]-dXij)**2
                stress = math.sqrt(stress)/np.linalg.norm(D['elist'])
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
        f0 = np.linalg.norm(D['dlist'])
        fX = 0
        for i in range(D['edges']):
            i1,i2 = D['elist'][i]
            Xij = X[i1]-X[i2]
            dij = np.linalg.norm(Xij)
            diffij = dij-D['dlist'][i]
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
        self.D['normalization'] = self.D['rms']*self.D['edges']
        self.N = D['nodes']; self.NN = D['edges']
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim
        
        self.f = lambda X, D=self.D, **kwargs : f(X,D,**kwargs)
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

    def gd(self, step_rule='mm', min_step=1e-4,**kwargs):
        if hasattr(self,'X') is False:
            self.initialize(title='automatic')
        if self.verbose > 0:
            print(f'  MDS.gd({self.title}):')
            #print(f'    step rule : {step_rule}')
            #print(f'    edge probability : {edge_probability}')
            print(f'    initial stress : {self.cost:0.2e}')

        Xi = self.subsample_generator(**kwargs)
        F = lambda X, xi=self.D : self.F(X,D=xi)
        self.X, H = gd.single(self.X,F,Xi=Xi,step_rule=step_rule,
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
    mds.gd(max_iter=200,min_grad=1e-10,min_step=1e-10,verbose=2,plot=True,
           **kwargs)
    mds.figureX(title='final embedding',colors=True)
    plt.show()

if __name__=='__main__':

    disk(N=30,step_rule='bb',edge_proportion=.9,lr=10)
