### Set up distances to be used by MPSE and other methods ###
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
import scipy.sparse.csgraph as csgraph
import itertools

### Functions to setup condensed distances from data ###

def setup_distances(data, shortest_path=False, min_distance=1e-4, **kwargs):
    """\
    Sets up condensed distances.

    Parameters
    ----------

    data : array
    Distance/dissimilarity data. Options:
    1) a condensed array containing distances (n_samples*(n_samples-1),)
    2) a square matrix containing distances (n_samples, n_samples)
    3) an array with features (length n_samples)

    shortest_path : boolean
    If True, alter distances by computing shortest path.

    kwargs
    ------

    metric : str
    If computing distances from an array of features, this is the metric to be
    passed to scipy.spatial.distance.pdist

    Returns
    -------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Condensed distances.
    """
    assert isinstance(data,np.ndarray)
    if len(data.shape) == 1:
        assert distance.is_valid_y(data)
        distances = data
    else:
        assert len(data.shape) == 2
        a, b = data.shape
        if b == a:
            distances = distance.squareform(data, checks=False)
        else:
            distances = distance.pdist(data)######,**kwargs)
    if shortest_path:
        distances = distance.squareform(distances)
        distances = csgraph.shortest_path(distances)
        distances = distance.squareform(distances, checks=False)
    if min_distance is not None:
        distances = np.maximum(distances,min_distance*np.max(distances))
    return distances

def setup_weights(distances, weights, max_weight=2.0, min_weight=1e-4):
    """\
    Sets up condensed weights array.

    Parameters
    ----------

    distances : array (n_samples*(n_samples-1),)
    Condensed distances.

    weights : None or str or callable or array
    Weights.
    """
    if isinstance(weights,str):
        if weights == 'reciprocal':
            weights = 1.0/distances
        else:
            sys.exit('weight type unknown')
    elif callable(weights):
        try:
            weights = weights(distance)
        except:
            weights = np.array([weights(dist) for dist in distances])
    elif isinstance(weights,np.ndarray):
        if len(weights) == distance.num_obs_y(distances):
            assert min(weights) >= 0
            weights = np.array([weights])
            weights = weights.T * weights
            weights = distance.squareform(weights, checks=False)
        assert distances.shape == weights.shape
    else:
        assert weights is None
        
    if weights is not None:
        if max_weight is not None:
            weights = np.minimum(weights,max_weight)
        if min_weight is not None:
            weights = np.maximum(weights,min_weight)
            
    return weights

def setup_distances_from_multiple_perspectives(data,data_args=None):
    """\
    Sets up condensed distances for each perspective.

    Arguments
    ---------

    data : list, length (n_perspectives)
    List containing distance/dissimilarity/feature data for each perspective.
    Each array can be of the following forms:
    1) A 1D condensed distance array
    2) A square distance matrix
    3) An array containing features
    ***4) A dictionary describing a graph

    data_args : dictionary (optional) or list
    Optional arguments to pass to distances.setup().
    If a list is passed, then the length must be the number of perspectives
    and each element must be a dictionary. Then, each set of distances will
    be set up using a different set of arguments.

    Returns
    -------

    condensed_distances : list, lenght (n_perspectives)
    List containing condensed distances for each perspective.
    """
    n_perspectives = len(data)

    if data_args is None:
        data_args = [{}]*n_perspectives
    elif isinstance(data_args, dict):
        data_args = [data_args]*n_perspectives
    else:
        assert isinstance(data_args,list)
        assert len(data_args) == n_perspectives
        for i in range(n_perspectives):
            if data_args[i] is None:
                data_args[i] = {}
            else:
                assert isinstance(data_args[i],dict)

    condensed_distances = []
    for i in range(n_perspectives):
        condensed_distances.append(setup_distances(data[i],**data_args[i]))

    return condensed_distances

### Distances to set up projections

### Function to return indeces in condensed distances for a given batch

def batch_indices(samples,n_samples):
    """\
    Returns the indices corresponding to pairs from given indices from the 
    condensed distance matrix list.

    Parameters
    ----------

    samples : list
    List with sample indices in batch.

    n_samples : int
    Total number of samples.

    Returns
    -------
    
    indices : list
    List of indices of condensed distance matrix containing distances involving
    samples in batch.
    """
    pairs = np.array(list(itertools.combinations(samples,2)))
    indices = n_samples*pairs[:,0]-pairs[:,0]*(pairs[:,0]+1)//2 + \
        pairs[:,1]-1-pairs[:,0]
    return indices

def distance_to_sample(distances,sample):
    return scipy.spatial.distance.squareform(distances)[sample]
