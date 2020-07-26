import os, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
sys.path.insert(1,'../..')
import mview

#book chapters are organized by volume, under 'data/' folder
folders = ['volume1/','volume2/','volume3/']

#count occurrences of characters in book:
occurrances = Counter()
for folder in folders:
    for filename in os.listdir("data/"+folder):
        f = open("data/"+folder+filename,'r')
        content = f.readlines()
        for line in content:
            name1, name2 = line.strip().split('\t')
            occurrances[name1] += 1
            occurrances[name2] += 1
        f.close()

#list of characters (in order of occurrances):
characters = [name for name,count in occurrances.most_common()]
appearences = [count for name,count in occurrances.most_common()]
n_characters = len(characters) #118
n_apperences = sum(appearences) #8066
#characters w/ at least 10 appearences: 0-40
#characters w/ at least 6 appearences: 0-48

counts = np.zeros((3,n_characters,n_characters),dtype=int)
for i in range(3):
    for filename in os.listdir("data/"+folders[i]):
        f = open("data/"+folders[i]+filename,'r')
        content = f.readlines()
        for line in content:
            name1, name2 = line.strip().split('\t')
            j= characters.index(name1)
            k = characters.index(name2)
            counts[i,j,k] += 1
            counts[i,k,j] += 1
        f.close()

#setup distance matrices
n_samples = 40
counts = counts[:,0:n_samples,0:n_samples]
distances = np.empty((3,n_samples,n_samples))
for i in range(3):
    distances[i] = 1.0 / np.maximum(0.5,counts[i])
    #distances[i] /= np.maximum(1,distances[i].sum(axis=1))[:,None]
    import scipy.sparse.csgraph as csgraph
    distances[i] = csgraph.shortest_path(distances[i])

def mds():
    mds = mview.MDS(distances[0],weights=None,
                    dim=2,verbose=2)
    mds.gd(batch_size=None,max_iter=30)
    mds.plot_embedding(labels=range(n_samples))
    mds.plot_computations()

def mpse():
    mv = mview.MPSE([distances[0],distances[1]],
                    data_args={'weights':'reciprocal'},verbose=2)
    mv.gd()
    mv.plot_embedding()
    mv.plot_images()
    mv.plot_computations()
    
mds()
mpse()
plt.show()
