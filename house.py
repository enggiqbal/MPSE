import numpy  as np
import matplotlib.pyplot as plt
import MPSE.mview as mview
import time





def main():
   
    path = 'MPSE/datasets/dataset_house/'
    y1 = np.genfromtxt(path+'house1.csv', delimiter = ',')
    y2 = np.genfromtxt(path+'house61.csv', delimiter = ',')
    y3 = np.genfromtxt(path+'house107.csv', delimiter = ',')
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(y1[:, 0], y1[:, 1], 'o')
    ax[1].plot(y2[:, 0], y2[:, 1], 'o')
    ax[2].plot(y3[:, 0], y3[:, 1], 'o')
    #plt.show()
    mv = mview.basic([y1, y2, y3], Q = None, verbose = 2)
    
    with open ('output.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")
    mv.plot_embedding()
    input()

main()
