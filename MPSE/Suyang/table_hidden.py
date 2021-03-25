import numpy  as np
import matplotlib.pyplot as plt
import time
from sys import path
from os.path import dirname as dir
import mpl_toolkits.mplot3d as plt3d
import nudged
import matplotlib.transforms as mtransforms
import math
import random
import copy
from datetime import datetime

path.append(dir(path[0]))
import mview as mview




def get_distance(y1):
    distance = []
    for i in range(len(y1)):
        temp = []
        for j in range(len(y1)):
            temp.append(math.sqrt(math.pow(abs(y1[i][0]-y1[j][0]),2) + math.pow(abs(y1[i][1]-y1[j][1]),2) ))
        distance.append(temp)
    return distance

def rotate(dom, r):
    a = np.cos(r)
    b = np.sin(r)
    rotate_mtx = np.array([[a, -b, 0.0], [b, a, 0.0], [0.0, 0.0, 1.0]],
                              float)
    result = np.dot(rotate_mtx, dom)

    return result

def do_plot(ax, Z, transform, y, weight, cost):
    '''
    im = ax.imshow(Z)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    '''
    #trans_data = transform + ax.transData

    print('Z:', Z)
    # display intended extent of the image
    ax.plot(y[:, 0], y[:, 1], ".", color = 'red')
    #ax.plot(Z[:, 0], Z[:, 1], ".", color = 'green')
    ax.set_xlabel('individual cost:'+ f'{cost}')
    #ax.plot(y[:, 0], y[:, 1], "o", color = 'red',
    #        transform=trans_data)
    #if weight is not None:
    '''
        if weight is not None, then the hidden points should label as color black
    '''
    for index in range(len(y)//2):
        if weight[index * (len(y) - 1)] == 0:
            #print(index, 'hidden------------------------------')
            ax.plot(y[index][0],y[index][1],".",color='black')

def check_hidden(obj, projection, ratio):
    """
         obj:    obj is the coordinate in each perspective
         projection: projection plane which is a list contains the coefficient
         ratio: how many points is visible

         return a dist which is the weight of the input
    """
    total = 0
    dist = []
    a = projection[0]
    b = projection[1]
    c = projection[2]
    for perspective in obj:
        each = []
        for points in perspective:
            d = (abs(a*points[0]+b*points[1]) / math.sqrt(a^2+b^2+c^2))
            total += d
            each.append(d)
        dist.append(each)
    
    avg = total / (len(obj) * len(obj[0]))
    
    """
            set points visibility
    """
    
    # set hidden point to sample 0 in perspective 1
    total = [x for x in range(len(obj[0])//2)]
    #sample = random.sample(total, int(len(obj[0])*ratio/2))

    y1 = obj[0]
    weights = []
    for i in range(3):
        each =  []
        for j in range((len(y1)-1)*len(y1)//2):      #weights
            each.append(1)
        weights.append(each)

    for perspective in range(len(dist)):
        for d in range(len(dist[perspective])//2):
        #for d in sample:
            ratio_temp = random.random()
            #if dist[perspective][d] < avg:                #if less thant avg distance then invisible
            if ratio_temp < ratio:                       
                index = d * (len(dist[perspective] ) -1)
                for i in range(index, index + len(dist[perspective]) - 1):
                    weights[perspective][i] = 0.0
    #print(total, (len(obj) * len(obj[0])), avg)
    #print(weights)
    weights = np.array(weights)
    return weights


def cal_dist(vec1, vec2):
    '''
        calculate the distance of the two vectors
    '''
    #sim = np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))
    #sim = sum(vec1*vec2) / (norm(vec1) * norm(vec2));
    sim = np.sqrt((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2 + (vec1[2]-vec2[2])**2 )
    return sim



def main():
    #2, 62, 108
    random.seed(datetime.now())
    path = './'
    y1 = np.genfromtxt(path+'1.csv', delimiter = ',')
    #x1 = plt.imread(path+'house.seq6.png')
    y2 = np.genfromtxt(path+'2.csv', delimiter = ',')
    #x2 = plt.imread(path+'house.seq70.png')
    y3 = np.genfromtxt(path+'3.csv', delimiter = ',')
    #x3 = plt.imread(path+'house.seq100.png')

    fig, ax = plt.subplots(1, 3)
    #ax[0].imshow(x1)
    ax[0].plot(y1[:, 0], y1[:, 1], 'o', color = 'red')
    for i in range(len(y1)):
        ax[0].annotate(i,y1[i])
    #ax[1].imshow(x2)
    ax[1].plot(y2[:, 0], y2[:, 1], 'o', color='red')
    for i in range(len(y2)):
        ax[1].annotate(i,y2[i])

    #ax[2].imshow(x3)
    ax[2].plot(y3[:, 0], y3[:, 1], 'o', color='red')
    for i in range(len(y3)):
        ax[2].annotate(i,y3[i])




    #projection_plane = ax+by+cz+d=0
    #distance = abs(a*x1+b*y1+c*z1) / sqrt(a^2+b^2+c^2)
    projection1 = [1, 1 , 1, 0]             #stands for x+ y +z =0 
    projection2 = [10, 10 , 10, 0]             #stands for 10x+ 10y +10z =0 
    projection3 = [1, 2 , 3, 0]             #stands for x+ 2y +3z =0 


    # set hidden point to sample 0 in perspective 1
    weights = []
    for i in range(3):
        each =  []
        for j in range((len(y1)-1)*len(y1)//2):      #weights
            each.append(1)
        weights.append(each)
    
    weights2 = copy.deepcopy(weights)
    weights2 = np.array(weights2)
    weights[0][0] = 0
    weights[1][0] = 0
    weights[2][0] = 1

    #y1 = np.array(get_distance(y1))
    #y2 = np.array(get_distance(y2))
    #y3 = np.array(get_distance(y3))

    weights = np.array(weights)

    ratio = 0.3
    weights = check_hidden([y1, y2, y3], projection3, ratio)
    #print(list(weights))

    #plt.show()
    mv = mview.basic([y1, y2, y3], weights = weights, smart_initialize=True, batch_size=10, max_iter=100, verbose = 2)

    mv2 = mview.basic([y1, y2, y3], weights = weights2, smart_initialize=True, batch_size=10, max_iter=100, verbose = 2)
    
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
    mv.plot_embedding()
    mv.plot_computations()
    mv.plot_images(labels=True)

    #choice is the two points that chose to be connected
    '''
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
    '''



    projections = []
    '''
    projections.append(mv.embedding @ mv.projections[0].T)
    projections.append(mv.embedding @ mv.projections[1].T)
    projections.append(mv.embedding @ mv.projections[2].T)
    '''
    projections = mv.images
    projections2 = mv2.images
    
    projection_matrices = mv.projections
    projection_matrices2 = mv2.projections
    #print('-----------projections--------------')
    #print(projection_matrices)
    #print(projection_matrices2)

    """
        calculate the normal vectors in each projection matrix
    """
    N = []
    N2 = []
    for index in range(len(projection_matrices)):
        N.append(np.cross(projection_matrices[index][0], projection_matrices[index][1]))
        N2.append(np.cross(projection_matrices2[index][0], projection_matrices2[index][1]))
    vec1 =  [abs(np.dot(N[0],N[1])) , abs(np.dot(N[0],N[2])), abs(np.dot(N[1],N[2]))]
    vec2 =  [abs(np.dot(N2[0],N2[1])) , abs(np.dot(N2[0],N2[2])), abs(np.dot(N2[1],N2[2]))]
    
    diff = cal_dist(vec1,vec2)

    #print('-------------')
    #print(vec1, vec2)
    print("two vector diff:", diff)

    with open('input.csv', 'w') as fd:
        for y in [y1, y2, y3]:
            for each in y:
                fd.write(','.join([str(each[0]), str(each[1])   ]) + '\n')
            fd.write('\n\n')

    with open('projection.csv', 'w') as fd:
        for projection in projections:
            for each in projection:
                fd.write(','.join([str(each[0]), str(each[1])   ]) + '\n')
            fd.write('\n\n')

    trans = []
    trans.append(nudged.estimate(y1, projections[0]))
    trans.append(nudged.estimate(y2, projections[1]))
    trans.append(nudged.estimate(y3, projections[2]))
    
    '''
    print('*******************')
    print('y1:', y1)
    print('projection[0]:', projections[0])
    test = nudged.estimate(y1, projections[0])
    r = test.transform(y1)
    print('transformation:', r)
    '''

    fig1, cx = plt.subplots(1, 3)

    #Z = [x1, x2, x3]
    Y = [y1, y2, y3]
    for i in range(len(trans)):
        each = trans[i]
        m = each.get_matrix()
        r = each.get_rotation()
        s = each.get_scale()
        t = each.get_translation()
        result = each.transform(Y[i])
        result  = np.array(result)
        #s = 1.0

        #print(rotate(Y[i], r))
        mse = nudged.estimate_error(each, Y[i], projections[i])
        print(i, 'error:', mse, 'scale:', s)
        #do_plot(cx[i], Z[i], mtransforms.Affine2D().
        #rotate(r).scale(s).translate(t[0], t[1]), projections[i], weights[i], mv.individual_cost[i])
        do_plot(cx[i], result, each, projections[i], weights[i], mv.individual_cost[i])

    plt.show()
    input()

main()
