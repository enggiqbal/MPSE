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

def stress_vs_miss(N=128,dim=2):
    num = 8
    misses = [1,2,4,8,16,32,64,128]
    fig, ax = plt.subplots(2,4,figsize=(12,6))
    axs = ax.flatten()
    fig.suptitle('normalized-stress / number-of-misplaced-nodes')
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(top=0.88)

    X = misc.disk(N,dim=dim)
    D = distance_matrix(X,X)
    vis = mds.MDS(D,dim=dim)

    stress = np.empty(num)
    for i in range(num):
        X_misplaced = X.copy()
        X_misplaced[-misses[i]::] = misc.disk(misses[i],dim=dim)
        stress[i] = mds.stress_function(X_misplaced,D,estimate=False)
        vis.initialize(X0=X_misplaced)
        vis.figureX(title=f'{stress[i]:0.2e} / {misses[i]}',
                    ax=axs[i])
    plt.show()

def stress_vs_estimate(N=128,dim=2):
    noise2signal = 10**np.arange(-2.5,0.5,0.5); num1=len(noise2signal)
    average_neighbors = [2,8,32,128,512]; num2=len(average_neighbors)
    its = 10

    X = misc.disk(N,dim=dim)
    D = distance_matrix(X,X)
    signal_std = np.std(X)
    noise = np.sqrt(noise2signal)*signal_std
    vis = mds.MDS(D,dim=dim)
    for i in range(num1):
        X_noisy = X+np.random.randn(N,dim)*noise2signal[i]
        true_stress = mds.stress_function(X_noisy,D,estimate=False)
        print(f'exact stress : {true_stress:0.2e}')
        for j in range(num2):
            print(f'  average neighbors : {average_neighbors[j]}')
            stress = 0
            stresses = []
            for k in range(its):
                stresses.append(mds.stress_function(
                    X_noisy,D,estimate=average_neighbors[j]))
            print(f'    average stress : {np.average(stresses):0.2e} '+\
                  f'[{abs(np.average(stresses)-true_stress)/true_stress:0.2e}]')
            print(f'    standard deviation : {np.std(stresses):0.2e} '+\
                  f'[{np.std(stresses)/true_stress:0.2e}]')
            print(f'    minimum stress : {min(stresses):0.2e} '+\
                  f'[{abs(min(stresses)-true_stress)/true_stress:0.2e}]')
            print(f'    maximum stress : {max(stresses):0.2e} '+\
                  f'[{abs(max(stresses)-true_stress)/true_stress:0.2e}]')
        
if __name__=='__main__':

    ### Quantification of normalized stress ###
    #stress_vs_noise(N=1024,dim=2)
    #stress_vs_miss(N=1024,dim=2)
    stress_vs_estimate(N=1024,dim=2)
