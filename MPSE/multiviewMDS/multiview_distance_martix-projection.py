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


import sys
#sys.path.append('../dataset')
#import graph_similarity_matrix as gsm
import data as mdata


from multiview_core import multiview

alpha=0.001
steps=2000
eps=1e-9
stopping_eps=0.1
dim=3
outputpath="../html3Dviz/"
if  len(sys.argv)!=8:
    print("uses: multiview_distance_matrix.py nameofdataset datapath1 datapath2 datapath3 learning_rate maxsteps")
    sys.exit()




name_data_set=sys.argv[1]
dpath1=sys.argv[2]
dpath2=sys.argv[3]
dpath3=sys.argv[4]

alpha= float(sys.argv[5])
steps=int(sys.argv[6])
outputpath=sys.argv[7]

D1=mdata.get_matrix(dpath1)
D2=mdata.get_matrix(dpath2)
D3=mdata.get_matrix(dpath3)
#D1=D1[0:10,0:10]
#D2=D2[0:10,0:10]
#D3=D3[0:10,0:10]

P1=np.random.rand(4,1)
P2=np.random.rand(4,1)
P3=np.random.rand(4,1)


print("number of data points",len(D1))
A=np.random.rand(len(D1)*dim,1)

mview=multiview(D1, D2, D3, dim, eps)
pos1,costs,P1,P2,P3=mview.multiview_mds_projection(A, P1, P2, P3, steps,alpha, stopping_eps,outputpath,name_data_set)

pos1=pos1.reshape(int(len(pos1)/dim),dim)


#write to file
jsdata ="var points="+ str(  pos1.tolist()) + ";"
file_path=outputpath+name_data_set +"_pos.js"
f=open(file_path,"w")
f.write(jsdata)
f.close()
np.savetxt(outputpath+name_data_set+"_pos.csv", pos1, delimiter=",")

#X,Y,Z=pos1.T[0], pos1.T[1], pos1.T[2]
#fig = plt.figure(1)
#ax = plt.axes(projection='3d')
#plt.title("3D View")
#plt.xlabel("X")
#plt.ylabel("Y")

print("position js saved at:", file_path)

print("position saved at:",outputpath+name_data_set+"_pos.csv")



#ax.scatter3D(X, Y, Z, c='red', cmap='Greens')

#x=np.arange(0, len(costs), dtype=float)
#w = np.cos(X)
#plt.figure(0)
#plt.title("Steps Cost")
#plt.xlabel("Steps")
#plt.ylabel("Cost")

#plt.plot(x, costs)

#plt.show()
