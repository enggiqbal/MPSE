import numpy  as np
import matplotlib.pyplot as plt
import time
from sys import path
from os.path import dirname as dir
import math
path.append(dir(path[0]))
import mview as mview
import sys




def main():
    #2, 62, 108
    filename = sys.argv[1]
    num = int(sys.argv[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with open (filename, 'r') as fd:
        for each in fd:
            data = each.split(',') 
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])
            #ax.scatter(x, y, z, c= 'cyan', alpha = 1.0, edgecolor = 'b')
            ax.scatter(x, y, z, color= 'blue')
    
    matrix = np.genfromtxt(filename, delimiter = ',') 

    for i in range(1, num+1):
        theta = math.pi/i
        a = np.array([[math.cos(theta),math.sin(theta),0],[0,0,1]])
        filename = str(i)+'.csv'
        projection = matrix @ a.T
        with open(filename, 'w') as fd:
            for each in projection:
                fd.write(','.join([str(each[0]), str(each[1])   ]) + '\n')



    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #mv.plot_embedding()
    #mv.plot_computations()
    #mv.plot_images(labels=True)
    plt.show()
    input()

main()
