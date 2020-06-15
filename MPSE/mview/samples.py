import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance

import misc, setup, multigraph, gd, projections, mds, tsne, plots

def 123():
    X = np.genfromtxt('samples/123/123.csv',delimiter=',')
    X1 = np.genfromtxt('samples/123/1.csv',delimiter=',')
    X2 = np.genfromtxt('samples/123/2.csv',delimiter=',')
    X3 = np.genfromtxt('samples/123/3.csv',delimiter=',')
    proj = projections.PROJ()
    Q = proj.generate(number=3,method='cylinder')
    return [X1,X2,X3], (X,Q)
    
