### Compute dissimilarity matrices for the florentine families dataset ###

import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

sys.path.insert(1,'../../..')
import mview

#file and pandas object with data:
#path = os.path.join(sys.path[0],
 #                   'florentine_families_relations_matrix_sample.csv')
path = 'florentine_families_relations_matrix_sample.csv'
with open(path) as csvfile:
    df = pd.read_csv(csvfile)

#list with attributes:
attributes = df.columns[2::]
# 'marriage', 'business', 'partnerships', 'bankemployment', 'realestate',
# 'patronage', 'loan', 'friendship', 'mallevadori'

#number of times each attribute occurs:
attributes_count = np.sum(df.values[:,2::],0)
#157, 58, 29, 14, 12, 44, 87, 17, 31

#list of families and number of times each family occurs:
families,families_count = np.unique(df.values[:,0:2],return_counts=True)
families = list(families)

#number of times a given family has each attribute, as either actor
counts = pd.DataFrame(np.zeros((len(families),len(attributes))),
                      index=families, columns=attributes)
for i in df.index:
    name1 = df['actor1surname'][i]; name2 = df['actor2surname'][i]
    counts.loc[name1] += df.loc[i][attributes]
    counts.loc[name2] += df.loc[i][attributes]

#similarity matrices for each attribute (counts):
similarities = np.zeros((len(attributes),len(families),len(families)))
for i in df.index:
    fam1 = df['actor1surname'][i]; i1 = families.index(fam1)
    fam2 = df['actor2surname'][i]; i2 = families.index(fam2)
    for j in range(len(attributes)):
        if df[attributes[j]][i]==1:
            similarities[j,i1,i2] += 1
            similarities[j,i2,i1] += 1

#definition of combined perspectives and similarities matrices
groups = ['marriage','loan','business','patronage']
s1 = similarities[0]#+similarities[7]
s2 = similarities[6]#+similarities[3]+similarities[4]
s3 = similarities[1]+similarities[2]
s4 = similarities[5]+similarities[8]
combined_similarities = np.array([s1,s2,s3,s4])

#definition of distances
with np.errstate(divide='ignore', invalid='ignore'):
    distances = np.true_divide(1.0,combined_similarities)
    distances[distances==np.inf]=0
    distances = np.nan_to_num(distances)
for i in range(4):
    distances[i] = shortest_path(distances[i],directed=False)

#indices of families in Medici network for marriage and loan attributes:
i = families.index('Medici')
perspectives = groups[0:2]
neighs1 = np.where(distances[0][i] != np.inf)[0]
neighs2 = np.where(distances[1][i] != np.inf)[0]
indices = list(set(list(neighs1)) & set(list(neighs2)))

#corresponding families and distance matrices
reduced_families = [families[i] for i in indices]
reduced_distances = distances[0:2][:,indices][:,:,indices]

#edges:
edges1 = []; edges2 = []
s1 = similarities[0]; s2 = similarities[6]
r1 = s1[indices][:,indices]
r2 = s2[indices][:,indices]
for i in range(len(reduced_families)):
    for j in range(i+1,len(reduced_families)):
        if r1[i,j] > 0:
            edges1.append([i,j])
        if r2[i,j] > 0:
            edges2.append([i,j])
edges = [edges1,edges2]

#indices corresponding to Medic and Strozzi:
medici = reduced_families.index('Medici')
strozzi = reduced_families.index('Strozzi')
labels = [None]*len(reduced_families)
labels[medici] = 'Medici'
labels[strozzi] = 'Strozzi'

def mds(i):
    mds = mview.MDS(reduced_distances[i],weights=lambda x:x**(-1),
                    dim=2,verbose=2)
    mds.gd(batch_size=None,max_iter=30)
    mds.plot_embedding(labels=labels,
                       colors=reduced_distances[i][medici],
                       axis=False,
                       title=' MDS embedding of '+perspectives[i]+' network'
                       ,edges=edges[i])
    mds.plot_computations()

def mpse():
    mv = mview.MPSE(reduced_distances,
                    data_args={'weights':lambda x:x**(-1)},verbose=2)
    mv.gd(bach_size=9,max_iters=50)
    mv.gd(max_iters=50,batch_size=16)
    mv.gd()
    mv.plot_embedding(labels=labels)#,axis=True)
    mv.plot_embedding(labels=labels,colors=reduced_distances[0][medici],
                      axis=True,edges=edges1)
    #mv.plot_embedding(labels=labels,colors=reduced_distances[1][medici],
    #                  axis=True,edges=edges2)
    mv.plot_computations()
    mv.plot_image(0,labels=labels,colors=reduced_distances[0][medici],
                  axis=False,title='MPSE embedding: marriage view',
                  edges=edges1)
    mv.plot_image(1,labels=labels,colors=reduced_distances[1][medici],
                  axis=False,title='MPSE embedding: loan view',
                  edges=edges2)
    ave_medici = np.linalg.norm(mv.embedding-mv.embedding[medici])/ \
        np.sqrt(len(reduced_families)-1)
    print('medici average',ave_medici)
    ave_strozzi = np.linalg.norm(mv.embedding-mv.embedding[strozzi])/ \
        np.sqrt(len(reduced_families)-1)
    print('strozzi average',ave_strozzi)
    
#mds(0)
#mds(1)
mpse()
plt.show()
