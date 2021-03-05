import sys
import matplotlib.pyplot as plt
import numpy as np
import setup_data
sys.path.insert(1,'../..')
import mview


def mds_all_features(n_samples=1000):
    features = setup_data.df[setup_data.df.columns[0:30]]\
        [0:n_samples].to_numpy()
    results = setup_data.results[0:n_samples]

    tsne = mview.TSNE(features,perplexity=20,dim=2,verbose=2)
    tsne.gd()
    tsne.plot_embedding(colors=results)
    plt.show()
    
    mds = mview.MDS(features,weights='reciprocal',dim=3,verbose=2)
    mds.gd(batch_size=20,max_iter=30)
    mds.plot_embedding()#colors=results)
    mds.plot_computations()
    plt.show()
    
def basic(groups = [0,1],n_samples=None):
    groups = [setup_data.group_names[group] for group in groups]
    data = []
    for group in groups:
        data.append(setup_data.generate_data(group,n_samples))
    results = setup_data.results[0:n_samples]
    for dat in data:
        mds = mview.MDS(dat,verbose=2)
        mds.gd(batch_size=50,max_iter=30)
        mds.plot_embedding(colors=results)
        mds.plot_computations()
    mv = mview.MPSE(data,verbose=2)
    mv.gd(batch_size=50,max_iter=30)
    mv.plot_embedding(colors=results)
    mv.plot_images(colors=results)
    mv.plot_computations()
    plt.show()

mds_all_features()
#basic(n_samples=1000)
