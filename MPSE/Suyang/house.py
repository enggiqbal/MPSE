import numpy  as np
import matplotlib.pyplot as plt
import time
from sys import path
from os.path import dirname as dir
import mpl_toolkits.mplot3d as plt3d

path.append(dir(path[0]))
import mview as mview




def main():
    #2, 62, 108
    path = '../datasets/dataset_house/'
    y1 = np.genfromtxt(path+'house2.csv', delimiter = ',')
    x1 = plt.imread(path+'house.seq2.png')
    y2 = np.genfromtxt(path+'house62.csv', delimiter = ',')
    x2 = plt.imread(path+'house.seq62.png')
    y3 = np.genfromtxt(path+'house108.csv', delimiter = ',')
    x3 = plt.imread(path+'house.seq108.png')
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(x1)
    ax[0].plot(y1[:, 0], y1[:, 1], 'o', color = 'red')
    for i in range(len(y1)):
        ax[0].annotate(i,y1[i])
    ax[1].imshow(x2)
    ax[1].plot(y2[:, 0], y2[:, 1], 'o', color='red')
    for i in range(len(y2)):
        ax[1].annotate(i,y2[i])

    ax[2].imshow(x3)
    ax[2].plot(y3[:, 0], y3[:, 1], 'o', color='red')
    for i in range(len(y3)):
        ax[2].annotate(i,y3[i])

    #plt.show()
    mv = mview.basic([y1, y2, y3], smart_initialize=True, batch_size=10, max_iter=700, verbose = 2)
    
    with open ('output.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")
    
    preds = np.array(mv.embedding)
    fig = plt.figure(figsize=plt.figaspect(.5))
    bx =  fig.add_subplot(1, 2, 2, projection='3d')
    surf = bx.scatter(preds[:, 0] ,\
                      preds[:, 1],\
                      preds[:, 2],\
                      c='cyan',\
                      alpha=1.0,\
                      edgecolor='b')
    preds = mv.embedding

    #choice is the two points that chose to be connected
    choice = [(0,1), (1,2), (23,24), (25,26), (8,9),(4,5),(5,6), (5,15),(15,17),
            (6,16), (16,18),(15,16),(17,18),(19,21),(19,20),(20,22), (21, 22),
            (27,28),(28,29),(10,11),(11,13),(11,12),(12,14),(13,14),(13,29),(14,29),
            (7,11)]
    for each in choice:
        start = each[0]
        end = each[1]
        xs = preds[start][0], preds[end][0]
        ys = preds[start][1], preds[end][1]
        zs = preds[start][2], preds[end][2]
        line = plt3d.art3d.Line3D(xs, ys, zs)
        bx.add_line(line)
    #bx.plot([preds[0]], [preds[1]] , color='blue')
    '''
    #bx.plot3D(preds[1:3, 0], preds[1:3, 1], preds[1:3, 2], color='blue')
    #bx.plot3D(preds[23:25, 0], preds[23:25, 1], preds[23:25, 2], color='blue')
    #bx.plot3D(preds[25:27, 0], preds[25:27, 1], preds[25:27, 2], color='blue')
    #bx.plot3D(preds[8:10, 0], preds[8:10, 1], preds[8:10, 2], color='blue')
    #bx.plot3D(preds[4:6, 0], preds[4:6, 1], preds[4:6, 2], color='blue')
    bx.plot3D(preds[5:7, 0], preds[5:7, 1], preds[5:7, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[5:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    bx.plot3D(preds[0:2, 0], preds[0:2, 1], preds[0:2, 2], color='blue')
    #bx.plot3D(preds[0:4, 0], preds[0:4, 1], preds[0:4, 1], color='blue')
    '''
    mv.plot_embedding()
    mv.plot_computations()
    mv.plot_images(labels=True)
    input()

main()
