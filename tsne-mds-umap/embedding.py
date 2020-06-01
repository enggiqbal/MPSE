from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import umap
from scipy.spatial import distance_matrix

import  seaborn as sn
import sys
sys.path.insert(0,'..')
import MPSE.mview as mview

def load_data(n):
    (trainX, labels), _ = mnist.load_data()
    trainX=np.reshape(trainX,(trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
    return trainX[0:n,:], labels[0:n]

def get_embedding(trainX,labels,fn,outfile):
    model = fn(n_components=2, random_state=0)
    pos = model.fit_transform(trainX)
    x=pos.T[0]
    y=pos.T[1]
    x=x/max(max(x), abs(min(x)))
    y=y/max(max(y), abs(min(y)))
    data = np.vstack((x,y, labels)).T
    df = pd.DataFrame(data=data, columns=("x", "y", "label"))
    draw_plot(df, 'x','y','label',outfile)
    return distance_matrix(df.values[:,0:2],df.values[:,0:2])

def draw_plot(data, x,y, l, outfile):
    df = pd.DataFrame(data=data, columns=(x, y, l))
    plt.clf()
    sn.FacetGrid(df, hue=l,  size=8).map(plt.scatter, x, y,s= 20).add_legend()
    plt.savefig(outfile+".png")

def get_MPSE(d1,d2,d3,labels):
    mv = mview.basic([d1,d2,d3], Q="standard", average_neighbors=2,  max_iter=500, verbose=2)
    data = np.vstack((mv.X.T, labels)).T
    df = pd.DataFrame(data=data, columns=("x", "y","z", "label"))
    draw_plot(df[["x","y","label"]], 'x','y','label',"1")
    draw_plot(df[["y","z","label"]], 'y','z','label',"2")
    draw_plot(df[["z","x","label"]], 'z','x','label',"3")
    return mv.X

if __name__ == "__main__":
    trainX, labels=load_data(100)
    tsne_dis=get_embedding(trainX,labels,TSNE,"tsne")
    mds_dis=get_embedding(trainX,labels,MDS,"mds")
    umap_dis=get_embedding(trainX,labels,umap.UMAP,"umap")

    mv=get_MPSE(tsne_dis,mds_dis,umap_dis,labels)


