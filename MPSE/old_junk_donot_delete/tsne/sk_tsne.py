import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
#import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad
import math
from sklearn.metrics import euclidean_distances, pairwise_distances
import autograd.numpy as np
from sklearn import manifold
import torch.nn.functional  as F
import matplotlib.lines as mlines
import pandas as pd
import umap

import sys
#sys.path.append('../dataset')
#import graph_similarity_matrix as gsm
import data as mdata



#dpath1='/Users/iqbal/multiview3d/dataset_3D/clusters_dataset/dist_2.csv'
dpath1='/Users/iqbal/multiview3d/dataset_3D/123_dataset_new/250/data_mat_1_250.csv'
#dpath1='/Users/iqbal/multiview3d/dataset_3D/sq_cir_tr_dataset/350/data_mat_sq_350.csv'

D1=mdata.get_matrix(dpath1)

#for i in range(1, 30):
#    for j in range(1 )
pr=20
ex=12
lr=1

filename="input_squire_perplexity_" + str(pr)
#tsne = manifold.TSNE(n_components=3,perplexity=10.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1, random_state=None, method='barnes_hut', angle=0.5)

tsne = manifold.TSNE(n_components=2,perplexity=pr, early_exaggeration=ex, learning_rate=lr, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1, random_state=None, method='barnes_hut', angle=0.5)
#pos = tsne.fit_transform(D1) #fit(similarities).embedding_


n_neighbors=5
filename="umap-input_2_n_neighbors_" + str(n_neighbors)

pos = umap.UMAP(n_neighbors=n_neighbors,
                      min_dist=0.3,
                      metric='euclidean').fit_transform(D1)



fig = plt.figure()
ax = plt.axes()
ax.grid(False)
plt.axis('off')
ax.scatter(pos.T[0], pos.T[1],  c='green', marker="x" ,  cmap='Greens');
plt.savefig(filename)
