import sys, os
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import projections, mds, mpse, misc
from multigraph import DISS
from projections import PROJ
from mds import MDS
from tsne import TSNE
from mpse import MPSE

def basic(data, data_args=None, fixed_projections=None,
          visualization_method='mds', smart_initialize=True,
          verbose=0,**kwargs):
    """\
    Basic function, finding mpse embedding/projections (the projections can be
    specified beforehand). It uses the mpse.MPSE() methods.

    Parameters
    ----------

    data : list, length (n_perspectives)
    List containing distance/dissimilarity/feature arrays (one array per
    perspective). Each array can be of the following forms:
    1) A 1D condensed distance array
    2) A square distance matrix
    3) An array containing features
    ***4) A dictionary describing a graph

    smart_initialize : boolean
    Start with a combined MDS embedding.

    kwargs
    ------

    data_args : dictionary or list (optional)
    Optional arguments to pass to distances.setup().
    If a list is passed, then the length must be the number of perspectives
    and each element must be a dictionary. Then, each set of distances will
    be set up using a different set of arguments.

    fixed_projections : None or list or array or string
    If set to None (default), projections are not assumed to be fixed.
    If a list of arrays, a combined array, or a string is given, then the 
    projections are fixed as given.

    visualization_method : string or list of strings
    Method must be 'mds' or 'tsne'.

    verbose : integer >= 0
    Level of verbose.

    fixed_embedding : array
    Fix the embedding array.

    initial_embedding : array
    Initial embedding (does not fix the embedding). If not given, then the
    initial is determined randomly.

    initial_projections : list of arrays or string
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

    Returns
    -------
    
    vis : MPSE object
    This object contains the results of the computations, along with plotting
    and further-computation methods.

    Some arguments are:
    vis.cost = current (final cost)
    vis.embedding = embedding
    vis.projections = projection parameters
    vis.computation_history = list with computation dictionaries
    """
    #old variables:
    if 'Q' in kwargs:
        fixed_projections = kwargs['Q']
        del kwargs['Q']
        
    vis = mpse.MPSE(data,verbose=verbose,fixed_projections=fixed_projections,
                    visualization_method=visualization_method,**kwargs)
    if smart_initialize is True and fixed_projections is None:
        vis.smart_initialize()
    if visualization_method == 'mds' and 'batch_size' not in kwargs:
        kwargs['batch_size'] = 10
    vis.gd(**kwargs)
    
    #old variables:
    vis.X = vis.embedding
    vis.Q = vis.projections
    vis.H = vis.computation_history
    
    return vis
