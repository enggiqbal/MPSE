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
    
     
    index=[l in [2,3] for l in labels]
    trainX=trainX[index]
    labels=labels[index]

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
    df[['x','y']].to_csv(outfile+'.csv')
    df[['label']].to_csv("label.csv")
    draw_plot(df, 'x','y','label',outfile)
    return distance_matrix(df.values[:,0:2],df.values[:,0:2])

def draw_plot(data, x,y, l, outfile):
    df = pd.DataFrame(data=data, columns=(x, y, l))
    plt.clf()
    sn.FacetGrid(df, hue=l,  size=8).map(plt.scatter, x, y,s= 20).add_legend()
    plt.savefig(outfile+".png")

def get_MPSE( labels):
    labels=pd.read_csv("label.csv")['label'].values 
    p=pd.read_csv("tsne.csv").values 
    d1=distance_matrix(p,p)
    p=pd.read_csv("umap.csv").values 
    d2=distance_matrix(p,p)
    # p=pd.read_csv("mds.csv").values
    # d3=distance_matrix(p,p)
 
    mv = mview.basic([d1,d2], Q="standard", average_neighbors=2,  max_iter=500, verbose=2)
    data = np.vstack((mv.X.T, labels.T)).T
    df = pd.DataFrame(data=data, columns=("x", "y","z", "label"))
    draw_plot(df[["x","y","label"]], 'x','y','label',"1")
    draw_plot(df[["y","z","label"]], 'y','z','label',"2")
    draw_plot(df[["z","x","label"]], 'z','x','label',"3")
    return mv.X

if __name__ == "__main__":
    trainX, labels=load_data(1000)
    tsne_dis=get_embedding(trainX,labels,TSNE,"tsne")
    # #mds_dis=get_embedding(trainX,labels,MDS,"mds")
    umap_dis=get_embedding(trainX,labels,umap.UMAP,"umap")

    mv=get_MPSE(labels)


