import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d


XYZ = np.array([[0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]])

CONN = np.array([[1,  4,  2],
                 [1,   3, 4],
                 [1,   6,  5],
                 [1,   2,  6],
                 [2,   8,   6],
                 [2,   4,   8],
                 [3,   8,   4],
                 [3,  7,  8],
                 [1,  7,   3],
                 [1,  5,  7],
                 [5,   8,  7],
                 [5,  6,  8]]) - 1

fig = plt.figure()
fig.set_size_inches(10,10)
ax = fig.add_subplot(111, projection='3d', aspect='equal')

#plot the nodes
for x, y, z in XYZ:
    ax.scatter(x, y, z, color='black', marker='s')

    #plot the lines
for ele, con in enumerate(CONN):
    for i in range(2):
        xs = XYZ[con[i]][0], XYZ[con[i+1]][0]
        ys = XYZ[con[i]][1], XYZ[con[i+1]][1]
        zs = XYZ[con[i]][2], XYZ[con[i+1]][2]
        line = plt3d.art3d.Line3D(xs, ys, zs)
        ax.add_line(line)
    xs = XYZ[con[0]][0], XYZ[con[2]][0]
    ys = XYZ[con[0]][1], XYZ[con[2]][1]
    zs = XYZ[con[0]][2], XYZ[con[2]][2]
    line = plt3d.art3d.Line3D(xs, ys, zs)
    ax.add_line(line)
plt.show()
input()
