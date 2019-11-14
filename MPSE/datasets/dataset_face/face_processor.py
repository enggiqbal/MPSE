import pandas as pd 
import numpy as np
import sys
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import preprocessing

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df



def draw_MDS(data, filename,titlearray):
    mds = MDS(n_components=2,dissimilarity="euclidean", verbose=2)
    pos = mds.fit(data).embedding_
    draw2d(pos,filename,titlearray)
     


def draw2d(points,filename,title):
    
    fig = plt.figure()
    ax = plt.axes()
    ax.grid(False)
    plt.axis('off')
    plt.title(title)
    ax.scatter(points.T[0], points.T[1],  c='black', marker="x" ,  cmap='Greens');
    plt.savefig(filename)






#filename='application_train_head.csv'
expname="face"
command_txt=""
for i in range(3, 4):
    print("dis", i)
    dst=pd.read_csv("face_D"+str(i)+".csv")
    #dst=normalize(dst)
    draw_MDS(dst,'dis'+expname+'_'+str(i)+'.png',"dis" + str(i))
    fnamme='dis'+expname+'_'+str(i)+'.csv'
    np.savetxt(fnamme,dst, fmt='%.6e', delimiter=',' )
    command_txt=command_txt + " ../dataset_{0}".format(expname) +"/"+ fnamme

print( 'python3 multiview_distance_martix.py face_123p1  '+ command_txt+' 0.001 10000 ../html3Dviz/ 1 3' )
print( 'python3 multiview_distance_martix.py face_123p2  '+ command_txt+' 0.001 10000 ../html3Dviz/ 2 3')



''''



dst2=pd.read_csv("face_D2.csv")
dst3=pd.read_csv("face_D3.csv")

dst2=normalize(dst2)
dst3=normalize(dst3)

draw_MDS(dst1,'dis'+expname+'_1.png',"dis1")
draw_MDS(dst2,'dis'+expname+'_2.png',"dis2")
draw_MDS(dst3,'dis'+expname+'_3.png',"dis3")


print( 'python3 multiview_distance_martix.py {0}_123p1 ../dataset_{0}/{0}_D1.csv ../dataset_{0}/{0}_D2.csv ../dataset_{0}/{0}_D3.csv 0.001 10000 ../html3Dviz/ 1 3'.format(expname))

print( 'python3 multiview_distance_martix.py {0}_123p2 ../dataset_{0}/{0}_D1.csv ../dataset_{0}/{0}_D2.csv ../dataset_{0}/{0}_D3.csv 0.001 10000 ../html3Dviz/ 2 3'.format(expname))


#python3 multiview_distance_martix.py projpeople_123p1 ../dataset_tabluar/dis_1.csv ../dataset_tabluar/dis_2.csv ../dataset_tabluar/dis_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeople1000_123p1 ../dataset_tabluar/dis1000_1.csv ../dataset_tabluar/dis1000_2.csv ../dataset_tabluar/dis1000_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeoplesmall_123p1 ../dataset_tabluar/dissmall_1.csv ../dataset_tabluar/dissmall_2.csv ../dataset_tabluar/dissmall_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeoplesmall2_123p1 ../dataset_tabluar/dissmall2_1.csv ../dataset_tabluar/dissmall2_2.csv ../dataset_tabluar/dissmall2_3.csv 0.001 10000 ../html3Dviz/ 1 3

#python3 multiview_distance_martix.py projpeoplesmall2_123p1 ../dataset_tabluar/dissmall2_1.csv ../dataset_tabluar/dissmall2_2.csv ../dataset_tabluar/dissmall2_3.csv 0.001 10000 ../html3Dviz/ 1 3
'''