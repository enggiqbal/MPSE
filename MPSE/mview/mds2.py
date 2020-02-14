### MDS implementation ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np

import misc, distances, multigraph, gd, plots

def dissimilarity_graph(D):
    """\
    Sets up dissimilarity graph to be used by MDS.
    
    Parameters:

    D : dictionary or numpy array
    Can be either i) a dictionary containing keys 'nodes', 'edges', 'distances',
    and 'weights'. D['nodes'] is a list of nodes (must be range(N) for some N as
    of now) and the rest are lists of the same size; ii) a dissimilarity 
    (square) matrix.

    Returns:

    D : dictionary
    Dissimilarity graph containing the following keys:
    'nodes' = list of nodes
    'edges' = list of edge pairs
    'distances' = list of distances corresponding to 'edges'
    'weights' = list of weights corresponding to 'edges'
    """
    if isinstance(D,np.ndarray):
        shape = D.shape; assert len(shape)==2; assert shape[0]==shape[1]
        D = multigraph.from_matrix(D)
        D['nodes'] = range(shape[0])

    assert 'nodes' in D
    assert 'edges' in D
    assert 'distances' in D

    if 'weights' not in D:
        D['weights'] = np.ones(len(D['edges']))

    D['distances'] = np.maximum(D['distances'],1e-4)
    
    d = D['distances']; w = D['weights']
    D['normalization'] = np.dot(w,d**2)
    D['rms'] = math.sqrt(D['normalization']/len(d))

    if 'colors' not in D:
        D['colors'] = None
        
    return D

def stress(X,D):
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

def stress_and_gradient(X,D,approx=None):
    """\
    Returns normalized MDS stress and gradient.
    
    Parameters:

    X : numpy array
    Positions/embedding array.

    D : dictionary
    Dictionary containing list of dissimilarities.

    approx : None or positive number
    If not None, approximates gradient/stress by only each edge with probability
    approx > 0.

    Returns:

    stress : float
    MDS stress at X.
    
    grad : numpy array
    MDS gradient at X.
    """
    e = D['edges']; d = D['distances']; w = D['weights']
    stress = 0; dX = np.zeros(X.shape)
    if approx is None:
        for n in range(len(e)):
            i,j = e[n]; Dij = d[n]; wij = w[n]
            Xij = X[i]-X[j]
            dij = np.linalg.norm(Xij)
            diffij = dij-Dij
            stress += wij*diffij**2
            dXij = wij*2*diffij/dij*Xij
            dX[i] += dXij
            dX[j] -= dXij
        stress /= D['normalization']
        dX /= D['normalization']
    else:
        assert isinstance(approx,numbers.Number); assert 0<approx<=1
        normalization = 0
        for n in range(len(e)):
            if np.random.rand() <= approx:
                i,j = e[n]; Dij = d[n]; wij = w[n]
                Xij = X[i]-X[j]
                dij = np.linalg.norm(Xij)
                diffij = dij-Dij
                stress += wij*diffij**2
                dXij = wij*2*diffij/dij*Xij
                dX[i] += dXij
                dX[j] -= dXij
                normalization += wij*Dij**2
        stress /= normalization
        dX /= normalization
    return math.sqrt(stress), dX

def approximate_stress_and_gradient(X,D,edge_probability=0.5):
    """\
    Returns approximate normalized MDS stress and gradient.
    
    Parameters:

    X : numpy array
    Positions/embedding array.

    D : dictionary
    Dissimilarity graph.

    edge_probability : positive number
    Approximates stress and gradient by using each edge with probability
    edge_probability > 0.

    Returns:

    stress : float
    Approximate MDS stress at X.
    
    grad : numpy array
    Approximate MDS gradient at X.
    """
    e = D['edges']; d = D['distances']; w = D['weights']
    stress = 0; dX = np.zeros(X.shape)
    p = edge_probability; assert isinstance(p,numbers.Number); assert p>0
    normalization = 0
    for n in range(len(e)):
        if np.random.rand() <= p:
            i,j = e[n]; Dij = d[n]; wij = w[n]
            Xij = X[i]-X[j]
            dij = np.linalg.norm(Xij)
            diffij = dij-Dij
            stress += wij*diffij**2
            dXij = wij*2*diffij/dij*Xij
            dX[i] += dXij
            dX[j] -= dXij
            normalization += wij*Dij**2
    stress /= normalization
    dX /= normalization
    return math.sqrt(stress), dX

class MDS(object):
    """\
    Class with methods to solve MDS problems.
    """
    def __init__(self, D, dim=2, verbose=0, title=''):
        """\
        Initializes MDS object.

        Parameters:

        D : dictionary or numpy array
        Either i) a dictionary with the lists of edges, distances, and weights 
        as described in dissimilarities.py or ii) a Can also be a dissimilarity
        matrix.

        dim : int > 0
        Embedding dimension.

        verbose : int >= 0
        Print status of methods in MDS object if verbose > 0.

        title : string
        Title assigned to MDS object.
        """
        if verbose > 0:
            print('+ mds.MDS('+title+'):')
        self.verbose = verbose; self.title = title

        self.D = dissimilarity_graph(D)
        self.N = len(D['nodes']); self.NN = len(D['edges'])
        
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim
        
        self.f = lambda X: stress(X,self.D)
        #self.df = lambda X: gradient(X,self.D)
        def F(X, approx=None):
            return stress_and_gradient(X,self.D,approx)
        self.F = F

        self.H = {}
        
        if verbose > 0:
            print(f'  number of points : {self.N}')
            print(f'  number of edges : {self.NN}')
            print(f'  (weighted) distance rms : {self.D["rms"]:0.2e}')
            print(f'  embedding dimension : {self.dim}')
            
    def initialize(self, X0=None, title='',**kwargs):
        """\
        Set initial embedding.

        Parameters:

        X0 : numpy array or None
        Initial embedding. If set to None, the initial embedding is produced 
        randomly using misc.initial_embedding().
        """
        if self.verbose > 0:
            print('- MDS.initialize('+title+'):')
            
        if X0 is None:
            X0 = misc.initial_embedding(self.N,dim=self.dim,
                                        radius=self.D['rms'],**kwargs)
            if self.verbose > 0:
                print('  method : random')
        else:
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.N,self.dim)
            if self.verbose > 0:
                print('  method : initialization given')
            
        self.X = X0
        self.update()
        
        self.X0 = self.X.copy()
        
        if self.verbose > 0:
            print(f'  initial stress : {self.cost:0.2e}')

    def update(self,H=None):
        self.cost = self.f(self.X)
        if H is not None:
            if bool(self.H) is True:
                H['cost'] = np.concatenate((self.H['cost'],H['cost']))
                H['steps'] = np.concatenate((self.H['steps'],H['steps']))
                H['iterations'] = self.H['iterations']+H['iterations']
            self.H = H        

    def forget(self):
        self.X = self.X0; self.H = {}
        self.update()

    ### Methods to update MDS embedding ###

    def gd(self, approximate=None, lr=10, max_iters=1000, min_step=1e-6,
           **kwargs):
        """\
        Updates MDS embedding using gradient descent.
        """
        if self.verbose > 0:
            print('- MDS.gd():')
            print('  approximate =',approximate)
        if approximate is None:
            F = lambda X: self.F(X)
        else:
            assert isinstance(approximate,numbers.Number)
            assert 0<approximate<=1
            F = lambda X: self.F(X,approx=approx)
        self.X, H = gd.gd(self.X,F,lr=lr,max_iters=max_iters,min_step=min_step,
                          **kwargs)
        self.update(H=H)
    
    def stochastic(self, approx=0.2, lr=10, max_iters=1000, min_step=1e-6,
                   **kwargs):
        """\
        Optimizes approximate stress function using gradient descent with a
        fixed learning rate.
        """
        F = lambda X: self.F(X,approx=approx)
        self.X, H = gd.mgd(self.X,F,lr=lr,max_iters=max_iters,min_step=min_step,
                           **kwargs)
        self.update(H=H)

    def agd(self, X0=None, **kwargs):
        F = lambda X: self.F(X)
        if self.verbose > 0:
            print('  method : adaptive gradient descent')
        self.X, H = gd.agd(self.X,F,**kwargs,**self.H)
        self.update(H=H)
        
    def optimize(self, agd=True, batch_size=None, batch_number=None, lr=0.01,
                 **kwargs):
        """\
        Optimize stress function using gradient-based methods. If batch size or
        number are given, optimization begins with stochastic gradient descent.
        If agd is set to True, optimization ends with adaptive gradient descent.
        """
        if self.verbose > 0:
            print('- MDS.optimize():')

        if batch_number is not None or batch_size is not None:
            F = lambda X: self.F(X,batch_number=batch_number,
                                 batch_size=batch_size)
            if self.verbose > 0:
                print('  method : stochastic gradient descent')
                if batch_number is None:
                    print(f'  batch size : {batch_size}')
                else:
                    print(f'  batch number : {batch_number}')
            self.X, H = gd.mgd(self.X,F,lr=lr,**kwargs)
            self.update(H=H)
        if agd is True:
            F = lambda X: self.F(X)
            if self.verbose > 0:
                print('  method : exact gradient & adaptive gradient descent')
            self.X, H = gd.agd(self.X,F,**kwargs,**self.H)
            self.update(H=H)

        if self.verbose > 0:
            print(f'  final stress : {self.cost:0.2e}')

    ### Plotting methods ###

    def figureX(self,title='mds embedding',edges=False,plot=True,ax=None):
        assert self.dim >= 2
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        if edges is True:
            edges = self.D['edges']
        else:
            edges = None
        colors = self.D['colors']
        plots.plot2D(self.X,edges=edges,colors=colors,ax=ax)
        if plot is True:
            plt.draw()
            plt.pause(1)

    def figureH(self,title='computations',plot=True,ax=None):
        assert hasattr(self,'H')
        if ax is None:
            fig, ax = plt.subplots()
        plots.plot_cost(self.H['cost'],self.H['steps'],title=title,ax=ax)
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

def example_disk(N=100,dim=2):
    print('\n***mds.example_disk()***')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y)
    
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1],c=colors)
    plt.title('original data')
    plt.draw()
    plt.pause(1)
    
    D = multigraph.from_coordinates(Y,colors=colors)
    title = 'basic disk example'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    mds.figureX(title='initial embedding')
    mds.gd(max_iters=50,approx=N/5,verbose=2)
    mds.agd(min_step=1e-6,verbose=2)
    mds.figureX(title='final embedding')
    mds.figureH()
    plt.show()

def test_gd_lr(N=100,dim=2):
    print('\n***mds.gd_lr()***')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y) 
    D = multigraph.from_coordinates(Y,colors=colors)
    title = 'recovering random coordinates for different learning rates'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    for lr in [100,10,1,.1]:
        mds.gd(lr=lr)
        mds.figure(title=f'lr = {lr}')
        mds.forget()
    plt.show()

def example_stochastic(N=100,dim=2):
    print('\n***mds.example_stochastic()***\n')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y)
    
    D = multigraph.from_coordinates(Y,colors=colors)

    title = 'recovering random coordinates from full dissimilarity matrix ' +\
            'using SGD, same learning rate, and different approx'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    for approx in [1.,.8,.6,.4,.2,.1]:
        mds.stochastic(verbose=1,lr=10.0,min_step=1e-6,
                       approx=approx,title=f'SGD using {approx} of edges')
        mds.figure(title=f'approx = {approx}, time = {mds.H["time"]:0.2f}')
        mds.forget()
    plt.show()
    
def example_weights(N=100,dim=2):
    print('\n***mds.example_weights()***\n')
    print('Here we explore the MDS embedding for a full graph for different'+
          'weights')
    title='MDS embedding for multiple weights'
    X = misc.disk(N,dim); colors = misc.labels(X)
    X0 = misc.disk(N,dim)
    
    D = multigraph.from_coordinates(X,colors=colors)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'absolute weights')

    multigraph.set_weights(D,scaling=.5)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'1/sqrt(Dij) weights')

    multigraph.set_weights(D,scaling=1)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'1/Dij weights')

    multigraph.set_weights(D,scaling=2)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'relative weights')

    plt.show()
    
def example_fewer_edges(N=100,dim=2):
    print('\n***mds.example_fewer_edges()***\n')
    print('Here we explore the MDS embedding for a full graph as far way edges'
          +'are removed')
    title='MDS embedding for multiple proportion of edges'
    X = misc.disk(N,dim); colors = misc.labels(X)
    D = multigraph.from_coordinates(X,colors=colors)
    X0 = misc.disk(N,dim)*.5
    for prop in [.99,.8,.6,.4,.2]:
        DD = multigraph.remove_edges(D,proportion=prop)
        mds = MDS(DD,dim=dim,verbose=1,title=title)
        mds.initialize(X0=X0)
        mds.stochastic(verbose=1,max_iters=300,approx=.99,lr=.5)
        mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
        mds.figure(title=f'proportion = {prop:0.1f}')
    plt.show()

def example_random_graph(N=100,dim=2):
    print('\n***mds.example_random_graph()***\n')
    print('Here we explore the MDS embedding for a random binomial graph with'+\
          'different edge probabilities.')
    for p in [0.01,0.05,0.1,1.0]:
        D = multigraph.generate_binomial(N,p)
        mds = MDS(D,dim=dim,verbose=1)
        mds.initialize()
        mds.stochastic(max_iters=50,approx=.6,lr=.5)
        mds.agd(min_step=1e-6)
        mds.figure(title=f'edge prob = {p:0.2f}')
    plt.show()

def disk_compare(N=100,dim=2): ###
    print('\n***mds.disk_compare()***')
    
    X = misc.disk(N,2); labels = misc.labels(X)
    
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.title('original data')
    plt.draw()
    plt.pause(0.1)
    
    D = distances.compute(X)
    
    mds = MDS(D,dim=dim,verbose=1,title='disk experiments',labels=labels)
    mds.initialize()
    mds.figureX(title='initial embedding')

    title = 'full gradient & agd'
    mds.optimize(algorithm='agd',verbose=2,label=title)
    mds.figureX(title=title)
    mds.figureH(title=title)

    mds.forget()
    title = 'approx gradient & gd'
    mds.approximate(algorithm='gd',verbose=2,label=title)
    mds.figureX(title=title)
    mds.figureH(title=title)

    mds.forget()
    title = 'combine'
    mds.approximate(algorithm='gd',verbose=2,label=title)
    mds.optimize(verbose=2,label=title,max_iters=10)
    mds.figureX(title=title)
    mds.figureH(title=title)
    plt.show()

def example_disk_noisy(N=100,dim=2):
    print('\n***mds.example_disk_noisy()***\n')
    noise_levels = [0.001,0.005,0.01,0.03,0.07,0.1,0.15,0.2,0.7,1.0]
    stress = []
    Y = misc.disk(N,dim)
    D = distances.compute(Y)
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        mds = MDS(D_noisy,dim,verbose=1,title=f'noise : {noise:0.2f}')
        mds.initialize()
        mds.optimize(algorithm='agd',max_iters=300,verbose=1)
        stress.append(mds.ncost)
    fig = plt.figure()
    plt.loglog(noise_levels,stress,'.-')
    plt.xlabel('noise level')
    plt.ylabel('stress')
    plt.title('Normalized MDS stress for various noise levels')
    plt.show()
    
def example_disk_dimensions(N=100):
    print('\n***mds.example_disk_dimensions()***\n')
    dims = range(1,11)
    stress = []
    for dim in dims:
        Y = misc.disk(N,dim)
        D = distances.compute(Y)
        mds = MDS(D,dim,verbose=1,label=f'dimension : {dim}')
        mds.initialize_Y()
        mds.optimize(algorithm='agd',max_iters=300)
        stress.append(mds.ncost)
    fig = plt.figure()
    plt.semilogy(dims,stress)
    plt.xlabel('dimension')
    plt.ylabel('stress')
    plt.title('Normalized MDS stress for various dimensions')
    plt.show()

### EMBEDDABILITY TESTS ###

def embeddability_dims(ax=None):
    print('\n**mds.embeddability_dims()')
    N=50
    ncost = []
    dims = list(range(2,50,5))
    #XX = misc.disk(N,20)
    XX = misc.box(N,20)
    for dim in dims:
        X = XX[:,0:dim]
        D = multigraph.coord2dict(X,weights='relative')
        mds = MDS(D,dim=2,verbose=1)
        mds.initialize()
        mds.optimize()
        ncost.append(mds.ncost)
    if ax is None:
        fig, ax = plt.subplots(1)
        plot = True
    else:
        plot = False
    ax.plot(dims,ncost)
    if plot is True:
        plt.show()

def embeddability_noise(ax=None):
    print('\n**mds.embeddability_noise()')
    N=50
    ncost = []
    noise_list = [0]+10**np.arange(-4,0,0.5)
    X = misc.disk(N,4)
    DD = distances.compute(X)
    for noise in noise_list:
        D = DD*(1+np.random.randn(N,N)*noise)
        mds = MDS(D,dim=4,verbose=1)
        mds.initialize()
        mds.optimize()
        ncost.append(mds.ncost)
    if ax is None:
        fig, ax = plt.subplots(1)
        plot = True
    else:
        plot = False
    ax.semilogx(noise_list,ncost)
    if plot is True:
        plt.show()
if __name__=='__main__':

    example_disk(N=100)
    #test_gd_lr()
    #example_approx(N=100)
    #example_weights(N=100,dim=2)
    #example_fewer_edges(N=60,dim=2)
    #example_random_graph(N=100,dim=2)
    
    #disk_compare(N=100)
    #example_disk_noisy(50)
    #example_disk_dimensions(50)

    #embeddability_dims()
    #embeddability_noise()