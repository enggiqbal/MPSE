
import sys
import networkx as nx
import pygraphviz as pgv
import math
from networkx.readwrite import json_graph
from networkx.drawing.nx_agraph import write_dot



import numpy as np
#from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d as plt3d

dotpath='../dataset/total_graph.dot'

G=nx.MultiGraph()
def GetSimlarityMatrix(dotpath):
    nodes_index={}
    G=nx.MultiGraph(pgv.AGraph(dotpath))
    SP=nx.all_pairs_shortest_path_length(G)
    n=len(G.nodes())
    count=0
    for node in G.nodes():
        nodes_index[node]=count
        count=count+1
    M = np.zeros(shape=(n,n))
    for x in SP:
        for y in x[1]:
            i=nodes_index[x[0]]
            j=nodes_index[y]
            M[i][j]=x[1][y]
    return M, G, nodes_index


seed = np.random.RandomState(seed=3)
 
similarities,  G, nodes_index = GetSimlarityMatrix(dotpath)

mds = manifold.MDS(n_components=3, max_iter=300, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1, verbose=3)
pos = mds.fit(similarities).embedding_


fig = plt.figure()
ax = plt.axes(projection='3d')

X,Y,Z=pos.T[0], pos.T[1], pos.T[2]

color=[]
for n in G.nodes():
    if G.node[n]["dept"]=="Mathematics":
        c='r'
    if G.node[n]["dept"]=="Systems and Industrial Engr":
        c='g'
    if G.node[n]["dept"]=="Computer Science":
        c='b'
    color.append(c)
#print(color)

ax.scatter3D(X, Y, Z, c=color, cmap='Greens');


#for e in G.edges():


for e in G.edges():
#    print(G[e[0]][e[1]])
    for e1 in G[e[0]][e[1]]:
        #print(G[e[0]][e[1]][e1])
        if G[e[0]][e[1]][e1]["edgetype"]=="dept":
            continue
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
