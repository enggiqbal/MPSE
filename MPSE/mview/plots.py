import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle; import csv
import numpy as np

def plot3D(X,save=False,perspectives=None,edges=None,colors=None,
           title=None,axis=False,ax=None):
        
    if ax is None:
        fig= plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        plot = True
    else:
        plot = False

    if perspectives is not None:
        q = perspectives
        #max0=np.max(X[:,0])
        #max1=np.max(X[:,1])
        #max2=np.max(X[:,2])
        #maxes = np.array([max0,max1,max2])
        #min0=np.min(X[:,0])
        #min1=np.min(X[:,1])        
        #min2=np.min(X[:,2])
        #mines = np.array([min0,min1,min2])
        for k in range(len(q)):
            #ind1 = np.argmax(np.max(q[k]/maxes))
            #ind2 = np.argmax(np.max(q[k]/mines))
            #if q[k][ind1] >= q[k][ind2]:
             #   ind = ind1
             #   m = abs(maxes[ind])
            #else:
             #   ind = ind2
              #  m = abs(mines[ind])
            ind = np.argmax(np.sum(q[k]*X,axis=1))
            m = np.linalg.norm(X[ind])/np.linalg.norm(q[k])
            ax.plot([0,m*q[k][0]],[0,m*q[k][1]],[0,m*q[k][2]],'-',linewidth=3,
                    color='black')
    N = len(X)
    if edges is not None:
        for i in range(N):
            for j in range(i+1,N):
                if edges[i,j] > 0:
                    ax.plot([X[i,0],X[j,0]],
                            [X[i,1],X[j,1]],
                            [X[i,2],X[j,2]],'-',
                            linewidth=0.25,color='blue')
                        
    ax.scatter3D(X[:,0],X[:,1],X[:,2],c=colors)
    ax.set_aspect(1.0)
    if axis is False:
        ax.set_axis_off()
    if title is not None:
        ax.title.set_text(title)
        
    if plot is True:
        plt.draw()
        plt.pause(0.1)

    if save is not False:
        assert isinstance(save,str)
        args = {
            'plot_type' : '3d',
            'X' : X,
            'perspectives' : perspectives,
            'edges' : edges,
            'colors' : colors,
            'title' : title,
            'axis' : axis
            }
        with open("plots/"+save+".pkl","wb") as handle:
            pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename,ax=None):
    assert isinstance(filename,str)
    with open("plots/"+filename+".pkl", 'rb') as handle:
        args = pickle.load(handle)
    plot_type = args['plot_type']
    if plot_type == '3d':
        X = args['X']
        perspectives = args['perspectives']
        edges = args['edges']
        colors = args['colors']
        title = args['title']
        axis = args['axis']
        plot3D(X,perspectives=perspectives,edges=edges,colors=colors,
                title=title,axis=axis,ax=ax)

def load_hull(filename):
    X = np.loadtxt(open("../visualhull/examples/123/"+filename+
                        ".csv","rb"),delimiter=',')
    plot3D(X)

def load_hull2(filename):
    X = np.loadtxt(open("../visualhull/examples/xyz/"+filename+
                        ".csv","rb"),delimiter=',')
    plot3D(X)

if __name__=='__main__':
    load('mlv3')
    load('mlf3')
    load('mlv2')
    load('mlf2')
    load_hull2('prata_1000_xyz')
    load_hull('gravitas_one_1000_123')
    load_hull('ceviche_one_500_123')
    plt.show()
