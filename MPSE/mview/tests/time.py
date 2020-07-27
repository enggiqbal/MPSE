import sys
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1, '../')
import misc, projections, mpse

def time(n_samples,n_perspectives,fixed_projections=False,
         trials=50, best=40):
    proj = projections.PROJ()
    times = []
    for k in range(trials):
        X = misc.disk(n_samples,dim=3)
        Q = proj.generate(number=n_perspectives,method='random')
        data = proj.project(Q,X)
        if fixed_projections:
            mv = mpse.MPSE(data,fixed_projections=Q)
        else:
            mv = mpse.MPSE(data)
        #mv.smart_initialize()
        mv.gd(batch_size=20,max_iter=1000,min_cost=1e-3,min_grad=1e-8)
        if mv.cost < 1.5e-3:
            times.append(mv.time)
        #mv.plot_computations()
        #mv.plot_embedding()
        #mv.plot_images()
        #plt.show()
    print(len(times),np.average(np.sort(times)[0:best]))


def comparison():
    n_samples = np.array(10**np.arange(1.5,4.01,.5),dtype=int); N=len(n_samples)
    n_perspectives = [2,3,4,5]; K=len(n_perspectives)
    trials = 2
    best = 3

    timef = np.empty((N,K,trials))
    timev = np.empty((N,K,trials))

    proj = projections.PROJ()
    
    for i in range(N):
        for j in range(K):
            for k in range(trials):
                X = misc.disk(n_samples[i],dim=3)
                Q = proj.generate(number=n_perspectives[j],method='random')
                data = proj.project(Q,X)
                X0 = misc.disk(n_samples[i],dim=3)
                
                mvf = mpse.MPSE(data,fixed_projections=Q,initial_embedding=X0)
                mvf.gd(batch_size=20,max_iter=500,min_cost=1e-4)
                timef[i,j,k] = mvf.time
                print(i,j,k,mvf.cost,mvf.time)
                mvf.plot_computations()
                plt.show()
                
                mvv = mpse.MPSE(data,initial_embedding=X0)
                mvv.gd(batch_size=20,max_iter=500,min_cost=1e-4)
                timev[i,j,k] = mvv.time
                print(mvv.cost,mvv.time)
                mvv.plot_computations()
                plt.show()

#### RUN #####

time(100,3,fixed_projections=False)
#comparison()
