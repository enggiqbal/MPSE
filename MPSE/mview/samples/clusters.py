import numpy as np
import scipy.spatial.distance
import math
import itertools

def equidistant(n_samples, distance=1.0, noise=0.1, **kwargs):
    "condensed distance array for n_samples equidistant points"
    length = n_samples*(n_samples-1)//2 #length of condensed distance array
    distances = np.random.normal(distance,noise,length) #distance+noise
    return distances

def clusters(n_samples, n_clusters=2, inner_distance=1.0, outer_distance=2.0,
             noise=0.1, **kwargs):
    "condensed distance array for n_clusters clusters"
    length = n_samples*(n_samples-1)//2 #length of condensed distance array
    distances = np.random.normal(outer_distance,noise,length)
    colors = np.empty(n_samples, dtype=int)
    permutation = np.random.permutation(n_samples)
    size = math.ceil(n_samples/n_clusters) #cluster size
    for i in range(n_clusters):
        ia=i*size; ib=min((i+1)*size, n_samples)
        indices = np.sort(permutation[ia:ib])
        colors[indices] = i
        pairs = np.array(list(itertools.combinations(indices,2)))
        indices = n_samples*pairs[:,0]-pairs[:,0]*(pairs[:,0]+1)//2 + \
        pairs[:,1]-1-pairs[:,0]
        distances[indices] = np.random.normal(inner_distance,noise,len(indices))
    return distances, colors

def clusters2(n_samples, n_clusters=2, outer_distance=10.0, **kwargs):
    
    od = outer_distance
    if n_clusters == 1:
        centers = [[0,0]]
    elif n_clusters == 2:
        centers = [[0,0],[od,0]]
    elif n_clusters == 3:
        centers = [[0,0],[od,0],[od/2,od*3**(1/2)/2]]
    elif n_clusters == 4:
        centers = [[0,0],[od,0],[od,od],[0,od]]
    elif n_clusters == 5:
        od2 = od*2**(1/2)/2
        centers = [[0,0],[od2,od2],[od2,-od2],[-od2,-od2],[-od2,od2]]
    elif n_clusters == 6:
        centers = [[0,0],[0,od],[0,2*od],[od,2*od],[od,od],[od,0]]
    else:
        print('cluster2() not difined for this number of clusters')
            
    x = np.random.normal(0,1,(n_samples,2))
    colors = np.empty(n_samples, dtype=int)
    permutation = np.random.permutation(n_samples)
    size = math.ceil(n_samples/n_clusters)
    for i in range(n_clusters):
        ia=i*size; ib=min((i+1)*size, n_samples)
        indices = permutation[ia:ib]
        colors[indices] = i
        x[indices] += centers[i]

    return x, colors

def createClusters(numbPoints, numbPerspectives):
    "creates data set with 2 clusters for each perspective"
        
    retClusters = []
    labels = []

    for i in range(numbPerspectives):

        meanFirst = (0, 0)
        A = np.random.rand(2, 2)
        covFirst = [[0.5, -0.1], [-0.1, 0.5]] + np.dot(A, A.transpose())

        x = np.random.multivariate_normal(meanFirst, covFirst, numbPoints//2)
    
        meanSecond = (8 + np.random.randn(1)[0], 8 + np.random.randn(1)[0])
        A = np.random.rand(2, 2)
        covSecond = [[0.5, 0.1], [0.1, 0.5]] + np.dot(A, A.transpose())

        y = np.random.multivariate_normal(meanSecond, covSecond,
                                          (numbPoints - numbPoints//2))
        
        z = np.concatenate((x, y))
        perm = np.random.permutation(numbPoints)
        z = z[perm]

        retClusters.append(z)
        currlabels = [0]*numbPoints
        for i in range(numbPoints):
            if perm[i] >= numbPoints/2:
                currlabels[i] = 1
        labels.append(currlabels)

    return retClusters, labels
