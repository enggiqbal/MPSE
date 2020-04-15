#python3 mpse.py -d datasets/dataset_tabluar/data/dissimple1000_1.csv  datasets/dataset_tabluar/data/dissimple1000_2.csv -d3 datasets/dataset_tabluar/data/dissimple1000_3.csv

import argparse
import math
 
import numpy as np
import os, sys
import MPSE.mview as mview
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import euclidean_distances, pairwise_distances 
from scipy.spatial import distance_matrix

import time
def get_D(number_of_points,number_of_projections):
    D=[]
    for i in range(0,number_of_projections):
        points_2D=np.c_[np.random.rand(number_of_points), np.random.rand(number_of_points)]
        d=pairwise_distances(points_2D)
        D.append(d)
    return D


def get_D_3projections(N):
    X = np.random.randn(N,3) #positions in 3D
    Y1 = X[:,[1,2]] #projection of data into 2D, viewed from x-direction
    Y2 = X[:,[2,0]]
    Y3 = X[:,[0,1]]
    D1 = distance_matrix(Y1,Y1) #pairwise distance matrix of Y1
    D2 = distance_matrix(Y2,Y2)
    D3 = distance_matrix(Y3,Y3)
    D=[D1,D2,D3]
    return D


def project_exp(points,max_projection, expname,  average_neighbors = 4):
    f = open(expname+".csv", "a")
    totalRun=10
    s=np.arange(1,totalRun+1)
    rc=','.join([f'r%dc'%c for c in s ])
    rit=','.join([f'it%d'%c for c in s ])
    rt=','.join([f'r%dt'%c for c in s ])
    f.write("projection,avgtime,avgcost,success,failed,"+rc+","+rit+"," + rt+"\n" )
    f.close()
    max_iter=1000
    for projections in range(1, max_projection):
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        for i in range(0,totalRun):
            DD = {'nodes' : points,  'attributes' : projections}
            mv = mview.basic(DD, average_neighbors=average_neighbors, max_iter=max_iter, min_cost=0.0001 )
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
            if max_iter==mv.H['iterations']:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
        h=','.join([f'%0.6f' % c for c in costs ])
        iterations=','.join([f'%d' % c for c in iterations ])
        timehistory=','.join([f'%0.2f' % c for c in timeRec ])
        f = open(expname+".csv", "a")
        f.write(f'%d,%.2f,%.2f,%d,%d,%s,%s,%s\n' %(projections,sum(timeRec)/len(timeRec),sum(costs)/len(costs),successcount[0],successcount[1], h, iterations,timehistory) )
        f.close()
  


def neighbor_exp(points,max_neigbors, expname):
    f = open(expname+".csv", "a")
    s=np.arange(1,11)
    rc=','.join([f'r%dc'%c for c in s ])
    rit=','.join([f'it%d'%c for c in s ])
    rt=','.join([f'r%dt'%c for c in s ])
    f.write("neigbors,avgtime,avgcost,success,failed,"+rc+","+rit+"," + rt+"\n" )
    f.close()
    max_iter=1000
    for ne in range(1, max_neigbors):
        average_neighbors=2**ne
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        for i in range(0,10):
            #DD = {'nodes' : points,  'attributes' : projections}
            D=get_D_3projections(points)
            mv = mview.basic(D, Q="standard", average_neighbors=average_neighbors, max_iter=max_iter, min_cost=0.0001 )
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
            if max_iter==mv.H['iterations']:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
        h=','.join([f'%0.6f' % c for c in costs ])
        iterations=','.join([f'%d' % c for c in iterations ])
        timehistory=','.join([f'%0.2f' % c for c in timeRec ])
        f = open(expname+".csv", "a")
        f.write(f'%d,%.2f,%.2f,%d,%d,%s,%s,%s\n' %(2**ne,sum(timeRec)/len(timeRec),sum(costs)/len(costs),successcount[0],successcount[1], h, iterations,timehistory) )
        f.close()
  


 
def points_exp(max_points, expname,fixed, average_neighbors = 0):
    f = open(expname+".csv", "a")
    projection_set="standard"
    step=100
    T=int(max_points/step)+1
    s=np.arange(1,11)
    rc=','.join([f'r%dc'%c for c in s ])
    rit=','.join([f'it%d'%c for c in s ])
    rt=','.join([f'r%dt'%c for c in s ])
    f.write("datapoints,avgtime,avgcost,success,failed,"+rc+","+rit+"," + rt+"\n" )
    f.close()
    max_iter=1000
    for n in range(1, T):
        points=n * step
        f = open(expname+".csv", "a")
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        for i in range(0,10):
            D=get_D_3projections(points)
            if fixed:
                if average_neighbors:
                    mv = mview.basic(D, Q="standard", average_neighbors=average_neighbors, max_iter=max_iter, min_cost=0.0001 )
                else:
                    mv = mview.basic(D, Q="standard", max_iter=max_iter, min_cost=0.0001 )
            else:
                if average_neighbors:
                    mv= mview.basic(D, average_neighbors=average_neighbors,max_iter=max_iter, min_cost=0.0001)
                else:
                    mv= mview.basic(D, max_iter=max_iter, min_cost=0.0001 )
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
            if max_iter==mv.H['iterations']:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
        h=','.join([f'%0.6f' % c for c in costs ])
        iterations=','.join([f'%d' % c for c in iterations ])
        timehistory=','.join([f'%0.2f' % c for c in timeRec ])
        f.write(f'%d,%.2f,%.2f,%d,%d,%s,%s,%s\n' %(points,sum(timeRec)/len(timeRec),sum(costs)/len(costs),successcount[0],successcount[1], h, iterations,timehistory) )
        f.close()
  

def projection_exp_old(points,total_projections, expname):
    f = open(expname+".csv", "a")
    f.write("projections,avgtime,avgcost\n" )
    f.close()

    n=points
    for projections in range(3, total_projections):
        #D=get_D(n,projections+1)
        costs=[]
        timeRec=[]
        f = open(expname+".csv", "a")
        mv={}
        for i in range(0,3):
            DD = {'nodes' : n,  'attributes' : projections}
            print(DD)
            mv[i] = mview.basic(DD,average_neighbors=4, max_iter=1000 )
            
            costs.append(mv[i].cost)
            timeRec.append( mv[i].time)
        f.write(f'%d,%.2f,%.6f\n' %(projections,sum(timeRec)/len(timeRec),sum(costs)/len(costs)) )
        print(f'%d,%.2f,%.6f\n' %(projections,sum(timeRec)/len(timeRec),sum(costs)/len(costs)) )
        print(costs)
        f.close()
 
#points_exp(2000, "fixed", 1)
#points_exp(2000, "veriable", 0)
#points_exp(2000, "fixed_average_neighbors", 1, average_neighbors=4)
#points_exp(2000, "veriable_average_neighbors", 0, average_neighbors=4)
project_exp(100,20, "project_exp_2",  average_neighbors = 4)
#neighbor_exp(100,10, "neighbor_exp")
 