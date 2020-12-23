import numpy  as np
import matplotlib.pyplot as plt
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
import mview as mview
import nudged
import matplotlib.transforms as mtransforms
import sys

SKIP = 20


def do_plot(ax, Z, transform, y):
    im = ax.imshow(Z)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    # display intended extent of the image
    ax.plot(y[:, 0], y[:, 1], "o", color = 'red')
    #ax.plot(y[:, 0], y[:, 1], "o", color = 'red',
    #        transform=trans_data)

def sample(matrix):
    temp = []
    for i in range(len(matrix)):
        if(i%SKIP == 0):
            temp.append([matrix[i][0], matrix[i][1]])

    return np.array(temp)



def main():
   
    fileName = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
    Z = []
    Y = []

    matrices = []
    for i in range(begin, end+1):
        name = str(i) + '.csv'
        try:
            y = np.genfromtxt(name, delimiter = ',') 
            #y = sample(y)
            Y.append(y)
        except:
            print("generate from", name, "failed\n")
            continue
        print(y)
        print(name)
        matrices.append(y)


    mv = mview.basic(matrices, smart_initialize=True, batch_size=10, max_iter=700, verbose = 2)
   
    #print('mv.individual_cost:', type(mv.individual_cost), mv.individual_cost)
    with open ('all.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]).strip(), str(each[1]).strip(), str(each[2]).strip()]) +"\n")
    mv.plot_embedding(title = 'final embedding')
    mv.plot_computations()
    mv.plot_images(title ='projection planes', labels=True)


    projections = []
    projections.append(mv.embedding @ mv.projections[0].T)
    projections.append(mv.embedding @ mv.projections[1].T)
    projections.append(mv.embedding @ mv.projections[2].T)

    trans = []
    trans.append(nudged.estimate(Y[0], projections[0]))
    trans.append(nudged.estimate(Y[1], projections[1]))
    trans.append(nudged.estimate(Y[2], projections[2]))

    Z = Y           #because here's no image 
    '''
    fig1, cx = plt.subplots(1, 3)
    for i in range(len(trans)):
        cx[i].set_title('projections '+ str(i))
        each = trans[i]
        m = each.get_matrix()
        r = each.get_rotation()
        s = each.get_scale()
        t = each.get_translation()
        #s = 1.0


        #print(rotate(Y[i], r))
        mse = nudged.estimate_error(each, Y[i], projections[i])
        print(i, 'error:', mse, 'scale:', s)
        cx[i].set_xlabel(mv.individual_cost[i])
        do_plot(cx[i], Z[i], mtransforms.Affine2D().rotate(r).scale(s).translate(t[0], t[1]), projections[i])
    '''
    input()

main()
