
import sys
import networkx as nx
import pygraphviz as pgv
import math
from networkx.readwrite import json_graph
from networkx.drawing.nx_agraph import write_dot
import sys
sys.path.append('../dataset')
import graph_similarity_matrix as gsm
import numpy as np
#from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d as plt3d

#dotpath='../dataset/total_graph.dot'
dotpath='../dataset/game_of_thrones_consistent.dot'
similarities,  G, nodes_index=gsm.get_similarity_matrix(dotpath)


seed = np.random.RandomState(seed=3)
print(nx.info(G))
#similarities,  G, nodes_index = GetSimlarityMatrix(dotpath)

mds = manifold.MDS(n_components=3, max_iter=300, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_


fig = plt.figure()
ax = plt.axes(projection='3d')

X,Y,Z=pos.T[0], pos.T[1], pos.T[2]

color=[]

ax.scatter3D(X, Y, Z, c='r', cmap='Greens');


#for e in G.edges():


for e in G.edges():
#    print(G[e[0]][e[1]])
    for e1 in G[e[0]][e[1]]:
        Xs=[]
        Ys=[]
        Zs=[]

        Xs.append(pos[nodes_index[e[0]]][0])
        Ys.append(pos[nodes_index[e[0]]][1])
        Zs.append(pos[nodes_index[e[0]]][2])
        Xs.append(pos[nodes_index[e[1]]][0])
        Ys.append(pos[nodes_index[e[1]]][1])
        Zs.append(pos[nodes_index[e[1]]][2])


        line = plt3d.art3d.Line3D(Xs,Ys,Zs)
        ax.add_line(line)


#lc = LineCollection(segments)
#lc.set_array(similarities.flatten())
#lc.set_linewidths(0.5 * np.ones(len(segments)))
#ax.add_collection(lc)



plt.show()
