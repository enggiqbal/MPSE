from mpl_toolkits import mplot3d
from matplotlib import pyplot
from stl import mesh
import sys

SKIP = 20

def sample(matrix):
    temp = []
    for i in range(len(matrix)):
        if(i%SKIP == 0):
            temp.append([matrix[i][0], matrix[i][1], matrix[i][2]])

    return np.array(temp)


def main():
    fileName = sys.argv[1]
    output  = fileName.split('.')[0]



    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    your_mesh = mesh.Mesh.from_file(fileName)
    
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    i = 0
   
    with open(output+'.csv', 'w') as fd:
        for line in your_mesh.vectors:
            for each in line:
                if i % SKIP != 0:
                    continue
                fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")
                i += 1 

    scale = your_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()
    input()



main()

