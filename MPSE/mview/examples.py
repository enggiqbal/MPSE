import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance

import misc, setup, multigraph, gd, projections, mds, tsne, plots, mpse, samples
from mpse import MPSE

def compare_perplexity(perplexities=[30,300]):
    dataset = '123'
    data = samples.load(dataset)
    #D = [data['X']]*2
    D = [data['D'][2]]*2
    va = []
    for p in perplexities:
        va.append({'perplexity':p})
    mv = MPSE(D,visualization_method='tsne',
              visualization_args=va,
              colors=data['colors'],verbose=2)

    mv.gd()
        
    mv.plot_computations()
    mv.plot_embedding(title='final embeding')
    mv.plot_images()
    plt.draw()
    plt.pause(0.2)
    plt.show()
    
    return

def compare_mds_tsne(dataset='mnist', perplexity=30):
    data = samples.load(dataset)
    D = [data['X']]*2
    va = {'perplexity':perplexity}
    mv = MPSE(D,visualization_method=['mds','tsne'],
              visualization_args=va,
              colors=data['colors'],verbose=2)

    mv.gd()
        
    mv.plot_computations()
    mv.plot_embedding(title='final embeding')
    mv.plot_images()
    plt.draw()
    plt.pause(0.2)
    plt.show()
    
    return

if __name__=='__main__':
    compare_perplexity()
    #compare_mds_tsne()
