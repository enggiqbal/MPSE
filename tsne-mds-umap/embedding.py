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
    index=[l in [3,5,8] for l in labels]
    trainX=trainX[index]
    labels=labels[index]
    return trainX[0:n,:], labels[0:n]

def get_embedding(trainX,labels,fn,outfile):
    model = fn(n_components=2, random_state=0)
    
    pos = model.fit_transform(trainX)
    
    x=pos.T[0]
    y=pos.T[1]
    # x=x/max(max(x), abs(min(x)))
    # y=y/max(max(y), abs(min(y)))
    
    data = np.vstack((x,y, labels)).T
    df = pd.DataFrame(data=data, columns=("x", "y", "label"))
    df[['x','y']].to_csv(outfile+'.csv')
    df[['label']].to_csv("label.csv")
    draw_plot(df, 'x','y','label',outfile)
    return 0#distance_matrix(df.values[:,0:2],df.values[:,0:2])

def draw_plot(data, x,y, l, outfile):
    import pdb; pdb.set_trace()
    df = pd.DataFrame(data=data, columns=(x, y, l))
    plt.clf()
    #plt.scatter( x, y,s= 20)
    sn.FacetGrid(df, hue=l,  size=8).map(plt.scatter, x, y,s= 20).add_legend()
    plt.savefig(outfile+".png")

def get_MPSE( labels):
    labels=pd.read_csv("label.csv")['label'].values 
    p=pd.read_csv("tsne.csv") 
    p=p[['x','y']]
    d1=distance_matrix(p.values,p.values)

    p=pd.read_csv("umap.csv") 
    p=p[['x','y']]
    d2=distance_matrix(p.values,p.values)
    # p=pd.read_csv("mds.csv").values
    # d3=distance_matrix(p,p)
 
    mv = mview.basic([d1,d2], Q="standard", average_neighbors=2,  max_iter=500, verbose=2)
    data = np.vstack((mv.X.T, labels.T)).T
    df = pd.DataFrame(data=data, columns=("x", "y","z", "label"))
    draw_plot(df[["x","y","label"]], 'x','y','label',"1")
    draw_plot(df[["y","z","label"]], 'y','z','label',"2")
    draw_plot(df[["z","x","label"]], 'z','x','label',"3")
    return mv.X

def getText_data():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    p='/Users/iqbal/projects/devis/data.csv'
    dataset=pd.read_csv(p)
    dataset=dataset[dataset.is_public==True].reset_index()
 
    dataset=dataset.fillna("")
    dataset['txt']= dataset['name'].str.cat(dataset['description'],sep=" ")  
    vectorizer = CountVectorizer(min_df=5, stop_words='english')
    trainX = vectorizer.fit_transform(dataset.txt)
    category_labels = [str(int(x)) for x in dataset.average_rating]

    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(dataset.txt)

    return tfidf_word_doc_matrix, category_labels





if __name__ == "__main__":
    trainXX, labels=load_data(500)
    trainXy, labels=getText_data()
    trainX=np.asarray(trainXy.todense(), dtype=np.float)
    # import pdb; pdb.set_trace()
    #tsne_dis=get_embedding(trainX,labels,TSNE,"tsne")
    # #mds_dis=get_embedding(trainX,labels,MDS,"mds")
    #umap_dis=get_embedding(trainX,labels,umap.UMAP,"umap")
    tsne_dis=get_embedding(trainX,labels,TSNE,"tsneX")

    #mv=get_MPSE(labels)


