import pandas as pd 
import numpy as np
import sys
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt



def draw_MDS(data, filename,titlearray):
    mds = MDS(n_components=2)
    pos=mds.fit_transform(data)

    #pos = mds.fit(data).embedding_
    print(pos)
    draw2d(pos,filename,titlearray)
     


def draw2d(points,filename,titlearray):
    title= ', '.join(titlearray)
    fig = plt.figure()
    ax = plt.axes()
    ax.grid(False)
    plt.axis('off')
    plt.title(title)
    ax.scatter(points.T[0], points.T[1],  c='black', marker="x" ,  cmap='Greens');
    plt.savefig(filename)

 
d=np.arange(1,100)
 
d=d.reshape(-1,1)
dst = euclidean_distances(d)
print(dst)
draw_MDS(dst,"out.png","xx")



