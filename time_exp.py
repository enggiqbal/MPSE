#python3 mpse.py -d datasets/dataset_tabluar/data/dissimple1000_1.csv  datasets/dataset_tabluar/data/dissimple1000_2.csv -d3 datasets/dataset_tabluar/data/dissimple1000_3.csv

import argparse
import math
from MPSE.multiviewMDS import multiview_core 
from MPSE.multiviewMDS import data
import numpy as np
import os, sys
import MPSE.mview as mview
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import euclidean_distances, pairwise_distances

def get_D(n,p):
    D=[]
    for i in range(0,p):
        point1=np.c_[np.random.rand(n), np.random.rand(n)]
        d=pairwise_distances(point1)
        D.append(d)
    return D




 
eps=1e-9
stopping_eps=0.1
projections=3
max_iters=200
lr=0.01
 
n=10
projection_set="standard"
import time
ns=[]
ts=[]
cs=[]
T=500
print("datapoints,time,cost")
for i in range(1, T):
    n=i * 10
    D=get_D(n,projections)
    start_time = time.time()
    pos,_,costs=mview.MULTIVIEW0(D,Q= projection_set,X0=None, lr=lr,max_iters=max_iters,verbose=0)
    cost=costs[len(costs)-1]
    t= time.time() - start_time
    print(f'%d,%.2f,%.2f' %(n,t, cost), flush=True)
    ns.append(n)
    ts.append(t)
    cs.append(cost)
'''
# libraries
import matplotlib.pyplot as plt
import pandas as pd
df=pd.DataFrame({'datapoints': ns, 'time': ts, 'cost': cs })
plt.plot( 'datapoints', 'time', data=df, marker='o',  color='skyblue')
plt.savefig("exp1.png")
plt.plot( 'datapoints', 'cost', data=df, marker='o', color='olive')
plt.legend()
plt.savefig("exp2.png")
'''


'''
    D = D[0:args.projections]
    pos,projections,_,costs=mview.MULTIVIEW(D,X0=None,max_iters=args.max_iters,verbose=args.verbose)
'''