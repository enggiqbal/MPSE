import numpy as np
import scipy.spatial.distance
import math

def clusters(n_samples=1000, n_clusters=2, inner_distance=1,
             outer_distance=2, **kwargs):
    distances = np.full((n_samples,n_samples), float(outer_distance))
    size = math.ceil(n_samples/n_clusters) #cluster size
    for i in range(n_clusters):
        ia = i*size; ib = min((i+1)*size, n_samples) 
        distances[ia:ib,ia:ib].fill(inner_distance)
    distances = scipy.spatial.distance.squareform(distances, checks=False)
    distances += np.random.normal(0,0.01,len(distances))
    return distances
