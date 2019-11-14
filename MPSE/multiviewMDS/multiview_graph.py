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
import json
from networkx.readwrite import json_graph


import sys
sys.path.append('../dataset')
import graph_similarity_matrix_multiview as gsm


from multiview_core import multiview


dotpath="../dataset/game_of_thrones_consistent.dot"
graphname="game_of_thrones"


#dotpath="../dataset/total_graph.dot"
#graphname="collaboration"

outdir="../html3Dviz/"


alpha=0.003
steps=1200
eps=1e-9
stopping_eps=0.01
dim=3
D1,D2,D3, G, nodes_index=gsm.get_similarity_matrix(dotpath)
#D3=np.zeros((len(D1),len(D1)))
#D1[D1==0]=100
#D2[D2==0]=100
#D3[D3==0]=100
print(len(np.where( D1 == 0)))
A=np.random.rand(len(D1)*dim,1)

mview=multiview(D1, D2, D3, dim, eps)

pos1,costs=mview.multiview_mds(A,steps, alpha, stopping_eps,outdir,'collaboration')


pos1=pos1.reshape(int(len(pos1)/dim),dim)

np.savetxt("data_pos_" + graphname + ".csv", pos1, delimiter=",")


#write to file
jsdata ="var points="+ str(  pos1.tolist()) + ";"
file_path=outdir+"data_pos_" + graphname + ".js"
f=open(file_path,"w")
f.write(jsdata)
f.close()

d = json_graph.node_link_data(G)
#json.dump(d, open(outdir+'graph_'+graphname+'.json', 'w'))
j=open(outdir+'graph_'+graphname+'.json','w')
j.write("edges='" + str(d['links']).replace("\'","\"") + "';" )
j.write("nodes='" + str(d['nodes']).replace("\'","\"") + "';" )
j.close()

fig = plt.figure(1)
ax = plt.axes(projection='3d')
plt.title("3D View")
plt.xlabel("X")
plt.ylabel("Y")



X,Y,Z=pos1.T[0], pos1.T[1], pos1.T[2]
ax.scatter3D(X, Y, Z, c='red', cmap='Greens');




x=np.arange(0, len(costs), dtype=float)
w = np.cos(X)
plt.figure(0)
plt.title("Steps Cost")
plt.xlabel("Steps")
plt.ylabel("Cost")

plt.plot(x, costs)

plt.show()
