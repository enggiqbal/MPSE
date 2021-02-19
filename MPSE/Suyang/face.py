import numpy  as np
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sys import path
from os.path import dirname as dir

path.append(dir(path[0]))
import mview as mview




def main():
    #2, 62, 108
    #path = '../datasets/dataset_house/'
    path = './'
    data_path = '../datasets/dataset_face/'
    #y1 = np.genfromtxt(path+'face1_2d.csv', delimiter = ',')
    #x1 = plt.imread(data_path+'face1.png')
    y2 = np.genfromtxt(path+'face2_2d.csv', delimiter = ',')
    x2 = plt.imread(data_path+'face2.png')
    #y3 = np.genfromtxt(path+'face3_2d.csv', delimiter = ',')
    #x3 = plt.imread(data_path+'face3.png')
    #y4 = np.genfromtxt(path+'face4_2d.csv', delimiter = ',')
    #x4 = plt.imread(data_path+'face4.png')
    y5 = np.genfromtxt(path+'face5_2d.csv', delimiter = ',')
    x5 = plt.imread(data_path+'face5.png')
    y6 = np.genfromtxt(path+'face6_2d.csv', delimiter = ',')
    x6 = plt.imread(data_path+'face6.png')
    fig, ax = plt.subplots(1, 6)
    X = [ x2,  x5, x6]
    Y = [ y2,  y5, y6]
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                  'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                  'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                  'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                  'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                  'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                  'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                  'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                  'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    for i in range(3):
        ax[i].imshow(X[i])
        temp = Y[i]
        ax[i].plot(temp[:, 0], temp[:, 1], 'o')
        for j in range(len(temp)):
            ax[i].annotate(j, temp[j])

    '''
    ax[0].imshow(x1)
    ax[0].plot(y1[:, 0], y1[:, 1], 'o')
    for i in range(len(y1)):
        ax[0].annotate(i,y1[i])
    ax[1].imshow(x2)
    ax[1].plot(y2[:, 0], y2[:, 1], 'o')
    for i in range(len(y2)):
        ax[1].annotate(i,y2[i])
    B
    ax[2].imshow(x3)
    ax[2].plot(y3[:, 0], y3[:, 1], 'o')
    for i in range(len(y3)):
        ax[2].annotate(i,y3[i])
    '''
    #plt.show()
    mv = mview.basic(Y, smart_initialize=True, batch_size=10, max_iter=700, verbose = 2 )
    
    with open ('output.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]), str(each[1]), str(each[2])]) +"\n")


    preds = np.array(mv.embedding)
    print(preds)
    print('-----------')
    print(preds[0:17, 0], preds[0:17, 1], preds[0:17, 2])
    print('-----------')
    fig = plt.figure(figsize=plt.figaspect(.5))
    bx =  fig.add_subplot(1, 2, 2, projection='3d')
    surf = bx.scatter(preds[:, 0] ,\
                      preds[:, 1],\
                      preds[:, 2],\
                      c='cyan',\
                      alpha=1.0,\
                      edgecolor='b')
    for pred_type in pred_types.values():
        bx.plot3D(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                preds[pred_type.slice, 2], color='blue')
        print(preds[pred_type.slice, 0], preds[pred_type.slice, 1], preds[pred_type.slice, 2])

    mv.plot_embedding()
    mv.plot_computations()
    mv.plot_images(labels = True)
    #mv.figureY()
    #mv.figureH()
    input()

main()
