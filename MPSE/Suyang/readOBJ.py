from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from stl import mesh
import sys
import trimesh
import numpy  as np

scaleX = 0
scaleY = 0 
scaleZ = 0
SKIP = 10

def sample(matrix):
    temp = []
    for i in range(len(matrix)):
        if(i%SKIP == 0):
            temp.append([matrix[i][0], matrix[i][1], matrix[i][2]])

    return np.array(temp)


def readFile(filename):
    mesh = trimesh.load(filename)
    mesh.show()
    points = mesh.sample(1024)

    return np.array(points)

def readFile_old(filename):
    temp = []
    with open(filename, 'r') as fd:
        fd.readline()
        for each in fd:
            each = each.strip().split()
            if len(each) < 4 or each[0] != 'v':
                continue
            x = float(each[1])
            y = float(each[2])
            z = float(each[3])
            temp.append([x, y, z])

    
    mesh = sample(np.array(temp))

    return mesh

def main():
    fileName = sys.argv[1]
    output  = fileName.split('.')[0]



    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    #your_mesh = mesh.Mesh.from_file(fileName)
    
    #axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    points = readFile_old(fileName)
    print(points)
    #mesh = sample(mesh)
    with open(output+'.csv', 'w') as fd:
        for each in points:
            axes.scatter(each[0], each[1], each[2], color= 'blue')
            fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")


    #scale = your_mesh.points.flatten(-1)
    #axes.auto_scale_xyz(scaleX, scaleY, scaleZ)
    plt.show()
    input()


if __name__ == '__main__':
    main()


