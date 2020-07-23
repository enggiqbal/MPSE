
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd 
# file_path='/Users/iqbal/projects/MPSE-web/MPSE/mview_examples/data/123/output/fixed_123.csv'
# df=pd.read_csv(file_path,header=None)
 
# X,Y,Z=df[0].values, df[1].values, df[2].values
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X, Y, Z, c='green', cmap='Greens')
# plt.title("3D View of 1-2-3")
# plt.show()
 



file_path='/Users/iqbal/projects/MPSE-web/MPSE/MPSE/html3Dviz/cluster_123p1_pos.csv'
df=pd.read_csv(file_path,header=None)
 
label='/Users/iqbal/projects/MPSE-web/MPSE/MPSE/html3Dviz/labels.js'
labels=np.array([[1,4,5],[2,4,5],[2,3,6],[2,4,5],[1,3,5],[1,3,6],[2,4,6],[1,4,6],[2,3,6],[2,3,5],[2,4,6],[2,4,5],[1,4,6],[2,4,5],[2,3,6],[1,3,6],[2,4,6],[1,3,6],[1,4,6],[1,4,6],[1,4,6],[2,3,5],[2,4,6],[2,4,6],[1,4,5],[1,4,6],[2,3,5],[2,4,6],[1,3,6],[1,3,5],[1,3,5],[1,4,5],[2,4,5],[2,4,6],[1,3,6],[1,3,5],[2,4,5],[1,4,5],[2,3,5],[2,4,6],[1,4,6],[2,4,5],[1,3,5],[1,3,5],[1,4,5],[2,3,6],[1,4,6],[1,3,5],[1,4,6],[1,3,5],[2,4,6],[1,4,6],[1,3,5],[1,3,5],[2,4,5],[2,4,5],[1,3,5],[2,3,5],[1,3,5],[2,3,5],[2,4,5],[1,3,5],[1,4,6],[2,4,5],[1,3,5],[2,3,5],[2,3,6],[2,4,6],[1,4,5],[2,3,5],[1,3,5],[1,3,5],[1,3,6],[2,3,6],[1,3,6],[2,4,6],[1,4,6],[2,4,5],[2,4,6],[2,3,6],[2,3,6],[2,3,5],[2,3,5],[2,4,6],[2,3,5],[1,3,6],[2,4,5],[2,4,6],[2,3,6],[2,3,5],[2,4,6],[1,3,6],[2,3,5],[1,3,6],[2,3,6],[2,4,5],[1,4,6],[1,3,5],[1,4,6],[1,3,5],[2,4,6],[2,4,6],[2,4,5],[1,4,6],[1,3,6],[2,3,5],[1,3,5],[1,4,5],[2,3,6],[2,4,6],[2,3,5],[2,3,6],[2,4,5],[1,3,5],[2,3,5],[1,4,6],[2,4,6],[1,3,6],[1,3,5],[1,3,5],[1,3,5],[1,4,6],[2,4,6],[2,4,6],[1,4,6],[1,3,6],[2,4,5],[1,4,5],[2,4,6],[1,3,6],[1,4,6],[2,3,5],[1,3,6],[2,4,6],[1,3,5],[2,4,6],[2,4,5],[1,4,6],[2,3,5],[1,4,5],[1,4,6],[2,3,6],[2,3,6],[1,4,5],[1,3,5],[2,4,5],[2,3,6],[1,3,5],[1,4,6],[1,4,6],[1,3,5],[1,3,5],[2,3,5],[2,3,5],[1,3,5],[1,4,6],[2,3,5],[2,3,5],[2,3,6],[1,4,6],[2,4,6],[1,4,6],[2,4,5],[1,4,6],[2,4,5],[1,3,6],[1,4,6],[2,3,6],[1,4,5],[2,4,5],[2,3,5],[1,3,5],[1,3,6],[2,3,6],[2,4,6],[2,3,5],[1,4,5],[1,4,5],[1,3,5],[2,4,6],[1,4,5],[2,4,6],[1,4,5],[2,3,6],[1,4,5],[2,4,5],[2,3,6],[2,3,6],[2,4,6],[1,4,6],[1,4,5],[1,3,5],[2,3,6],[1,3,5],[1,4,6],[1,3,6],[1,3,6],[2,3,5],[2,4,5],[1,3,5]])

color=[]
markers=[]
facecolors=[]
X,Y,Z=df[0].values, df[1].values, df[2].values
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
 
for i in range(len(df)):
    



    if (labels[i][1] == 3):
        color.append('green')
    else:
        color.append('red')


    if (labels[i][2] == 5):
        facecolors.append('none')
    else:
        facecolors.append(color[i])
      
    if labels[i][0] == 1:
        markers.append('o')
    else:
        markers.append('s')
    if (labels[i][2] == 5):
    #     plt.scatter(X[i], Y[i], Z[i], facecolors='none', edgecolors=color[i], marker=markers[i])
    # else: 
        ax.scatter(X[i], Y[i], Z[i],facecolors='none', edgecolors=color[i], marker=markers[i])
    else:
   # plt.scatter(X[i], Y[i], Z[i], facecolors='none', edgecolors='r')
        ax.scatter(X[i], Y[i], Z[i],  c=color[i], marker=markers[i])




plt.title("3D View of cluster")
plt.show()
#plt.savefig('3Dcluster.png')
