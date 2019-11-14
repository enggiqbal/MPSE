import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from numpy import genfromtxt
from matplotlib import cm
from scipy.spatial import Delaunay
pos1 = genfromtxt('/Users/iqbal/multiview3d/html3Dviz/123_12p2_pos.csv', delimiter=',')

fig = plt.figure(1)
ax = plt.axes(projection='3d')
plt.title("3D View")
plt.xlabel("X")
plt.ylabel("Y")

X,Y,Z=pos1.T[0], pos1.T[1], pos1.T[2]
ax.scatter3D(X, Y, Z, c='red', cmap='Greens');
#ax.plot_trisurf(X, Y, Z,  linewidth=0.1)
tri = Delaunay(np.array([X,Y]).T)
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap=plt.cm.Spectral)


plt.show()
