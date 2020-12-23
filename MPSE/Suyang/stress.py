import numpy  as np
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
import mview as mview


path = './'
y2 = np.genfromtxt(path+'face2_2d.csv', delimiter = ',')
y5 = np.genfromtxt(path+'face5_2d.csv', delimiter = ',')
y6 = np.genfromtxt(path+'face6_2d.csv', delimiter = ',')
result = np.genfromtxt('3d.csv', delimiter = ',')
data = [y2, y5, y6]
mv = mview.MPSE(data)
#print(mv.projections)
#print(result)
#Y = mv.projections
#Y = None
projections = []
projections.append(result @ mv.projections[0].T)
projections.append(result @ mv.projections[1].T)
projections.append(result @ mv.projections[2].T)
print(projections)
cost, individual_cost = mv.cost_function(X=result ,Q=mv.projections,Y = None)
print(cost, individual_cost)


