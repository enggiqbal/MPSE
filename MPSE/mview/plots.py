import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle; import csv
import numpy as np

def plot_cost(cost,steps=None,title='computations',plot=True,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        plot = False
    if steps is not None:
        ax.semilogy(steps, label='step size', linestyle='--')
    ax.semilogy(cost, label='cost',linewidth=3)
    ax.set_xlabel('iterations')
    ax.legend()
    ax.set_title(title)
    if plot is True:
        plt.draw()
        plt.pause(1.0)
        
def plot2D(Y,save=False,colors=None,edges=None,title=None,ax=None,plot=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        plot = False

    if edges is not None:
        for i,j in edges:
            ax.plot([Y[i,0],Y[j,0]],
                    [Y[i,1],Y[j,1]],'-',
                    linewidth=0.15,color='gray')
    ax.scatter(Y[:,0],Y[:,1],s=25,c=colors)
    ax.title.set_text(title)
        
    if plot is True:
        plt.draw()
        plt.pause(1)

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
        for k in range(len(q)):
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
    #ax.set_aspect(1.0)
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