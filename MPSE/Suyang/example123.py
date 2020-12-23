import numpy  as np
import matplotlib.pyplot as plt
import time
from sys import path
from os.path import dirname as dir

path.append(dir(path[0]))
import mview as mview




def main():
    #2, 62, 108
    #path = '../datasets/dataset_house/'
    path = './'

    y1 = np.genfromtxt(path+'spicy_rice_1000_1.csv', delimiter = ',')
    y2 = np.genfromtxt(path+'spicy_rice_1000_2.csv', delimiter = ',')
    y3 = np.genfromtxt(path+'spicy_rice_1000_3.csv', delimiter = ',')
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(y1[:, 0], y1[:, 1], 'o')
    for i in range(len(y1)):
        ax[0].annotate(i,y1[i])
    ax[1].plot(y2[:, 0], y2[:, 1], 'o')
    for i in range(len(y2)):
        ax[1].annotate(i,y2[i])
    ax[2].plot(y3[:, 0], y3[:, 1], 'o')
    for i in range(len(y3)):
        ax[2].annotate(i,y3[i])

    #plt.show()
    mv = mview.basic([y1, y2, y3], smart_initialize=True, batch_size=10, max_iter=700, verbose = 2)
    
    with open ('output.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")
    mv.plot_embedding()
    mv.plot_computations()
    mv.plot_images(labels=True)
    input()

main()
