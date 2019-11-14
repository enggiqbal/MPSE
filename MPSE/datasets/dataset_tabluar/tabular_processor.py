import pandas as pd 
import numpy as np
import sys
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def draw_MDS(data, filename,titlearray):
    mds = MDS(n_components=2,dissimilarity="euclidean")
    pos = mds.fit(data).embedding_
    draw2d(pos,filename,titlearray)
     


def draw2d(points,filename,titlearray):
    title= ', '.join(titlearray)
    fig = plt.figure()
    ax = plt.axes()
    ax.grid(False)
    plt.axis('off')
    plt.title(title)
    ax.scatter(points.T[0], points.T[1],  c='black', marker="x" ,  cmap='Greens');
    plt.savefig(filename)






#filename='application_train_head.csv'
#expname="simple"

filename='application_head_1000.csv'
expname="credit3_1000"
#comments

data=pd.read_csv(filename)
projections=[["FLAG_OWN_CAR",	"FLAG_OWN_REALTY",	"AMT_INCOME_TOTAL",	"NAME_INCOME_TYPE"],
["NAME_TYPE_SUITE",	"CNT_CHILDREN",	"NAME_FAMILY_STATUS",	"OCCUPATION_TYPE"],
["NAME_EDUCATION_TYPE",	"ORGANIZATION_TYPE"]]
items_to_factorize=["FLAG_OWN_CAR","OCCUPATION_TYPE","NAME_INCOME_TYPE","ORGANIZATION_TYPE","NAME_FAMILY_STATUS","FLAG_OWN_REALTY","NAME_TYPE_SUITE"]
#simple version
projections=[[	"AMT_INCOME_TOTAL"],[	"CODE_GENDER"], ["NAME_EDUCATION_TYPE"]]
items_to_factorize=["CODE_GENDER"]

print(data["NAME_EDUCATION_TYPE"].unique())

col_to_keep= projections[0]+projections[1]+projections[2]

col_to_drop=set(list(data.columns)) - set(col_to_keep)
data=data.drop(col_to_drop, axis=1)
data=data.dropna()
data=data[data["NAME_EDUCATION_TYPE"]!='Lower secondary'].reset_index()

print(data.shape)
df=data.copy()
columnsTitles=[	"AMT_INCOME_TOTAL",	"CODE_GENDER","NAME_EDUCATION_TYPE"]
df=df.reindex(columns=columnsTitles)
a = np.asarray(df)
np.set_printoptions(threshold=sys.maxsize)
#np.savetxt( expname+'_label.csv', a,fmt='%s', delimiter=',' )
#np.save( expname+'_label.csv', data )
label_file_name=expname+'_label.js'
f=open(label_file_name,"w")
labels_data="var labels="+np.array2string(a, precision=6, separator=',', suppress_small=True)+";"
f.write(labels_data)
f.close()


edu_order={'Lower secondary':0,'Secondary / secondary special':1 , 'Incomplete higher':2 , 'Higher education':3 } 
data["NAME_EDUCATION_TYPE"]=[ edu_order[data["NAME_EDUCATION_TYPE"][x]] for x in range(0, len(data["NAME_EDUCATION_TYPE"]))]
 
for i in items_to_factorize:       
    data[i]=pd.factorize(data[i])[0]


 
p1 = data[projections[0]].copy()
p1=p1.values
p1=p1.T[0]
#p1.sort()
import matplotlib.pyplot as plt
import pdb; pdb.set_trace() 
plt.plot(p1) # Plot list. x-values assumed to be [0, 1, 2, 3]
plt.show()
 

data=(data-data.mean())/data.std()

p1 = data[projections[0]].copy()
p2 = data[projections[1]].copy()
p3 = data[projections[2]].copy()


 
dst1 = euclidean_distances(p1)
dst2 = euclidean_distances(p2)
dst3 = euclidean_distances(p3)

from sklearn.preprocessing import normalize
 
dst1=normalize(dst1)
dst2=normalize(dst2)
dst3=normalize(dst3)
 
np.savetxt('dis'+expname+'_1.csv',dst1, fmt='%.6e', delimiter=',' )
np.savetxt('dis'+expname+'_2.csv',dst2, fmt='%.6e', delimiter=',' )
np.savetxt('dis'+expname+'_3.csv',dst3, fmt='%.6e', delimiter=',' )

draw_MDS(dst1,'dis'+expname+'_1.png',projections[0])
draw_MDS(dst2,'dis'+expname+'_2.png',projections[1])
draw_MDS(dst3,'dis'+expname+'_3.png',projections[2])

print( 'python3 multiview_distance_martix.py {0}_123p1 ../dataset_tabluar/dis{0}_1.csv ../dataset_tabluar/dis{0}_2.csv ../dataset_tabluar/dis{0}_3.csv 0.0001 10000 ../html3Dviz/ 1 3'.format(expname))



#python3 multiview_distance_martix.py projpeople_123p1 ../dataset_tabluar/dis_1.csv ../dataset_tabluar/dis_2.csv ../dataset_tabluar/dis_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeople1000_123p1 ../dataset_tabluar/dis1000_1.csv ../dataset_tabluar/dis1000_2.csv ../dataset_tabluar/dis1000_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeoplesmall_123p1 ../dataset_tabluar/dissmall_1.csv ../dataset_tabluar/dissmall_2.csv ../dataset_tabluar/dissmall_3.csv 0.001 10000 ../html3Dviz/ 1 3
#python3 multiview_distance_martix.py projpeoplesmall2_123p1 ../dataset_tabluar/dissmall2_1.csv ../dataset_tabluar/dissmall2_2.csv ../dataset_tabluar/dissmall2_3.csv 0.001 10000 ../html3Dviz/ 1 3

#python3 multiview_distance_martix.py projpeoplesmall2_123p1 ../dataset_tabluar/dissmall2_1.csv ../dataset_tabluar/dissmall2_2.csv ../dataset_tabluar/dissmall2_3.csv 0.001 10000 ../html3Dviz/ 1 3
# python3.6 mview_hpc.py credit4 dataset_tabluar/data/discredit3_1000_1.csv dataset_tabluar/data/discredit3_1000_2.csv dataset_tabluar/data/discredit3_1000_3.csv  0.001 10000 ../html3Dviz/ 1 3  