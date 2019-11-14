import numpy as np 
points=np.load('123/computed123.npy')
np.savetxt("computedpoints.csv",points,delimiter=",")

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
X,Y,Z=points.T[0], points.T[1], points.T[2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='green', cmap='Greens')
plt.title("3D View")
plt.show()
plt.savefig('books_read.png')




points=np.load('123/computed1.npy')
points=np.load('123/computed2.npy')
points=np.load('123/computed3.npy')
fig = plt.figure()
ax = plt.axes()
ax.grid(False)
plt.axis('off')
ax.scatter(points.T[0], points.T[1],  c='green', marker="x" ,  cmap='Greens');
plt.savefig("computed3.png")




