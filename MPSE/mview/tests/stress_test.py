import sys
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from scipy.spatial import distance_matrix

sys.path.insert(1, '../')
import misc, multigraph, mds

### Quantification of normalized stress ###

def stress_vs_noise(N=128,dim=2):
    num = 6
    noise2signal = 10**np.arange(-2.5,0.5,0.5)
    fig, ax = plt.subplots(2,3,figsize=(9,6))
    axs = ax.flatten()
    fig.suptitle('noise-to-signal-ratio / normalized-stress')
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(top=0.88)

    X = misc.disk(N,dim=dim)
    signal_std = np.std(X)
    noise = np.sqrt(noise2signal)*signal_std
    D = distance_matrix(X,X)
    vis = mds.MDS(D,dim=dim)

    stress = np.empty(num)
    for i in range(num):
        X_noisy = X+np.random.randn(N,dim)*noise2signal[i]
        stress[i] = mds.stress_function(X_noisy,D,estimate=False)
        vis.initialize(X0=X_noisy)
        vis.figureX(title=f'{noise2signal[i]:0.2e} / {stress[i]:0.2e}',
                    ax=axs[i])
    plt.show()
    return

def stress_vs_shuffle(N=128,dim=2):
    num = 6
    noise2signal = 10**np.arange(-2.5,0.5,0.5),
    fig, ax = plt.subplots(2,3,figsize=(9,6))
    axs = ax.flatten()
    fig.suptitle('normalized-stress / noise-to-signal')
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(top=0.88)

    X = misc.disk(N,dim=dim)
    signal_std = np.std(X)
    noise = noise2signal*signal_std
    D = distance_matrix(X,X)
    vis = mds.MDS(D,dim=dim)

    stress = np.empty(num)
    for i in range(num):
        X_noisy = X+np.random.randn(N,dim)*noise2signal[i]
        stress[i] = mds.stress_function(X_noisy,D,estimate=False)
        vis.initialize(X0=X_noisy)
        vis.figureX(title=f'{stress[i]:0.2e} / {noise2signal[i]:0.2e}',
                    ax=axs[i])
    plt.show()
        
if __name__=='__main__':

    ### Quantification of normalized stress ###
    #stress_vs_noise(N=1024,dim=2)
    stress_vs_shuffle(N=1024,dim=2)
