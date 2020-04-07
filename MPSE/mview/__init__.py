import sys, os
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import projections, mds, mpse

def MDS(D,dim=2,X0=None,batch_number=None,batch_size=10,lr=0.01,max_iters0=200,
        max_iters=200,verbose=0,plot=False,title='MDS solution',labels=None,
        **kwargs):
    """\
    MDS function. It looks for embedding X by 1) stochastic gradient descent
    with given batch number/size and learning rate, repeating computation with
    a smaller learning rate if computation is unstable, iteratively until a
    stable solution is found; 2) adaptive gradient descent on full gradient.

    Parameters:

    D : numpy array
    Distance or dissimilarity matrix

    dim : int > 0
    Embedding dimension.

    X0 : None or numpy array.
    Optional initial parameters

    batch_number : None or int
    Number of batches used in stochastic gradient descent

    batch_size : int
    If batch_number is None, size of batches used in stochastic gradient 
    descent.

    lr : number > 0
    Initial learning rate used in stochastic gradient descent (iteratively 
    reduced if algorithm fails).

    max_iters0 : int
    Maximum number of iterations in stochastic gradient descent.

    max_iters : int
    Maximum number of iterations in adaptive gradient descent.

    verbose : 0 or 1 or 2
    Level of verbose. 0 for none, 1 for minimal, 2 for step-by-step.

    plot : boolean
    Return plot of computation history and final embedding if True.
    
    title : string
    Title used through computation (for verbose and plot purposes).

    labels : array-like
    Labels for positions/nodes (used to add color in plots).

    Returns:

    X : numpy array
    Final embedding/solution.

    cost : number
    Final cost.

    cost_history : numpy array
    Array containing cost history through computation.
    """
    vis = mds.MDS(D,dim=dim,verbose=verbose,title=title,labels=labels)
    vis.initialize(X0=X0)
    vis.optimize(batch_number=batch_number,batch_size=batch_size,lr=lr,
                 max_iters=max_iters0,verbose=verbose)
    #vis.optimize(max_iters=max_iters,verbose=verbose)
    if plot is True:
        vis.figure(title=title)
        plt.show()
    return vis.X, [], vis.cost, vis.H['cost']

def MULTIVIEW0(D,dimX=3,dimY=2,Q='same',X0=None,batch_number=None,batch_size=10,
               lr=0.01,max_iters0=200,max_iters=200,verbose=0,plot=False,
               title='MULTIVIEW solution',labels=None,**kwargs):
    """\
    MULTIVIEW0 function, finding multiview-mds embedding with fixed projections.
    It looks for embedding X by 1) if X0 is not give, initializing X0 by solving
    for the MDS embedding of the averaged distances; 2) stochastic gradient 
    descent with given batch number/size and learning rate, repeating 
    computation with a smaller learning rate if computation is unstable, 
    iteratively until a stable solution is found; 3) adaptive gradient descent 
    on full gradient. 

    Parameters:

    D : numpy array
    Distance or dissimilarity matrix

    dimX : int > 0
    Embedding dimension.

    dimY : int > 0
    Projection dimension.

    Q : string
    Choice for set of projections. The options are:
        same : all the projections are the 1st standard projection
        standard : the standard projections
        cylinder : projections orthogonal to last set of axes.
        orthogonal : random orthogonal projection
        normal : random projection with entries from normal distribution
        uniform : random projection with entries from uniform distribution

    X0 : None or numpy array.
    Optional initial parameters

    batch_number : None or int
    Number of batches used in stochastic gradient descent

    batch_size : int
    If batch_number is None, size of batches used in stochastic gradient 
    descent.

    lr : number > 0
    Initial learning rate used in stochastic gradient descent (iteratively 
    reduced if algorithm fails).

    max_iters0 : int
    Maximum number of iterations in stochastic gradient descent.

    max_iters : int
    Maximum number of iterations in adaptive gradient descent.

    verbose : 0 or 1 or 2
    Level of verbose. 0 for none, 1 for minimal, 2 for step-by-step.

    plot : boolean
    Return plot of computation history and final embedding if True.
    
    title : string
    Title used through computation (for verbose and plot purposes).

    labels : array-like
    Labels for positions/nodes (used to add color in plots).

    Returns:

    X : numpy array
    Final embedding/solution.

    cost : number
    Final cost.

    cost_history : numpy array
    Array containing cost history through computation.
    """
    
    assert isinstance(D,list)
    
    persp = perspective.Persp(dimX=dimX,dimY=dimY)
    if Q in ['same','standard','cylinder']:
        persp.fix_Q(special=Q,number=len(D))
    elif Q in ['orthogonal','normal','uniform']:
        persp.fix_Q(random=Q,number=len(D))
    else:
        sys.exit('Incorrect option for Q')
            
    vis = multiview.Multiview(D,persp,verbose=verbose,title=title,labels=labels)
    vis.setup_visualization(visualization='mds')
    vis.initialize_X(X0=X0,method='mds',batch_number=batch_number,
                    batch_size=batch_size,lr=lr,max_iters=max_iters0)

    vis.optimize_X(batch_number=batch_number,batch_size=batch_size,lr=lr,
                   verbose=verbose,max_iters=max_iters0)
    #vis.optimize_X(max_iters=max_iters,verbose=verbose)

    if plot is True:
        vis.figureX(title=title)
        vis.figure(title=title)
        plt.show()
    return vis.X, vis.cost, vis.H['cost']

def MULTIVIEW(D,dimX=3,dimY=2,X0=None,batch_number=None,batch_size=10,
              max_iters=200,**kwargs):
    """\
    MULTIVIEW function, finding multiview-mds embedding with variable 
    projections. It looks for embedding X by 1) if X0 is not give, initializing 
    X0 by solving for the MDS embedding of the averaged distances; 2) coordinate
    stochastic gradient descent with given batch number/size and learning rate, 
    repeating computation with a smaller learning rate if computation is 
    unstable, iteratively until a stable solution is found; 3) coordinate 
    adaptive gradient descent on full gradients. 

    Parameters:

    D : numpy array
    Distance or dissimilarity matrix

    dimX : int > 0
    Embedding dimension.

    dimY : int > 0
    Projection dimension.

    X0 : None or numpy array.
    Optional initial parameters

    batch_number : None or int
    Number of batches used in stochastic gradient descent

    batch_size : int
    If batch_number is None, size of batches used in stochastic gradient 
    descent.

    lr : number > 0
    Initial learning rate used in stochastic gradient descent (iteratively 
    reduced if algorithm fails).

    max_iters0 : int
    Maximum number of iterations in stochastic gradient descent.

    max_iters : int
    Maximum number of iterations in adaptive gradient descent.

    verbose : 0 or 1 or 2
    Level of verbose. 0 for none, 1 for minimal, 2 for step-by-step.

    plot : boolean
    Return plot of computation history and final embedding if True.
    
    title : string
    Title used through computation (for verbose and plot purposes).

    labels : array-like
    Labels for positions/nodes (used to add color in plots).

    Returns:

    X : numpy array
    Final embedding/solution.

    cost : number
    Final cost.

    cost_history : numpy array
    Array containing cost history through computation.
    """
    X, Q, c, H = MPSE(D,d1=dimX,d2=dimY,X0=X0,max_iter=max_iters,**kwargs)
    return X, Q, c, H['costs']

def basic(D,Q=None,verbose=0,smart_initialize=False,**kwargs):
    """\
    Basic function, finding mpse embedding/projections (the projections can be
    specified beforehand). It uses the mpse.MPSE() methods.

    Parameters:

    D : dictionary or list or numpy arrays
    Distance or dissimilarity matrices.

    Q : None or string or list of numpy arrays
    If None, Q is not specified and optimization runs for both X and Q.
    If not None, then Q is fixed and given as specified.

    verbose : number
    Prints verbose if > 0.

    plot : boolean
    Return plots of computation history and final embedding if True.
    
    Returns:

    X : array
    Final embedding.

    Q : list of arrays
    Final projection parameters.

    cost : number
    Final normalized cost.
    
    H : dictionary
    Dictionary with computation stats, such as:
    H['cost'] = cost history
    H['step'] = step size history
    H['grad'] = gradient size history
    H['time'] = total computation time

    kwargs (all optional):

    X0 : array
    Initial embedding (does not fix the embedding). If not given, then the
    initial is determined randomly.

    Q0 : list of arrays or string
    Initial projection parameters. Only relevant if the projection parameters
    are not fixed. If not given, then the initial projection parameters are
    determined randomly.

    X : array
    It is also possible to fix X (and only optimize Q).

    max_iter = number of maximum iterations
    min_step = minimum step size stopping criterion
    max_step = maximum step size stopping criterion
    min_cost = minimum cost stopping criterion
    lr = initial learning rate (unnecessary for adaptive schemes)

    edge_probability = None or a number between 0 and 1. If given, stochastic
    gradient descent is used (edges are included in computation with given
    probability).
    """
    vis = mpse.MPSE(D,Q=Q,verbose=verbose,**kwargs)
    if smart_initialize is True:
        vis.smart_initialize(verbose=verbose)
    vis.gd(verbose=verbose,**kwargs)
    return vis
