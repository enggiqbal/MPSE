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
from tqdm import tqdm
import time
totalRun=10
max_iter=1000
min_cost=0.01
def get_D_3projections(N):
    X = np.random.randn(N,3) #positions in 3D
    Y1 = X[:,[0,1]] #projection of data into 2D, viewed from x-direction
    Y2 = X[:,[2,0]]
    Y3 = X[:,[1,2]]
    D1 = distance_matrix(Y1,Y1) #pairwise distance matrix of Y1
    D2 = distance_matrix(Y2,Y2)
    D3 = distance_matrix(Y3,Y3)
    D=[D1,D2,D3]
    return D

def write_header(expname):
    f = open('expdata/'+expname+".csv", "a")
    global totalRun 
    s=np.arange(1,totalRun+1)
    rc=','.join([f'r%dc'%c for c in s ])
    rit=','.join([f'it%d'%c for c in s ])
    rt=','.join([f'r%dt'%c for c in s ])
    f.write("points,neighbors,projection,projectioncount,avgtime,avgcost,avgsucctime, avgsucccost,success,failed,"+rc+","+rit+"," + rt+"\n" )
    f.close()

def write_row(expname,row):
    f = open('expdata/'+expname+".csv", "a")
    f.write( row )
    f.close()



def project_exp(points,max_projection, expname, average_neighbors,ptype):
    write_header(expname)
    global max_iter 
    global min_cost 
    global totalRun 
    for projections in tqdm(range(1, max_projection)):
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        successtimeandcost=[]
        successtimeandcost.append([])
        successtimeandcost.append([])
        diss = mview.DISS(points) #start method to generate dissimilarities
        diss.add_projections(attributes=projections,Q=ptype) #specify that you want attribute_number attributes generated from physical 3D-2D example, using cylinder projections
        Q = diss.Q #recover projection parameters to use along mview.basic (if using fixed projections, otherwise unnecessary)


        for i in range(0,totalRun):
            #DD = {'nodes' : points,  'attributes' : projections}
            mv = mview.basic(diss, Q=Q, average_neighbors=average_neighbors, max_iter=max_iter, min_cost=min_cost )
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
            if abs(max_iter - mv.H['iterations']) < 2:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
                successtimeandcost[0].append(mv.time)
                successtimeandcost[1].append(mv.cost)

 
        process_runs(expname,points,average_neighbors, ptype, successcount, timeRec,costs,iterations, successtimeandcost, projections)

def process_runs(expname,points,average_neighbors, projections, successcount, timeRec,costs,iterations, successtimeandcost, projectionCount):
    avgsucctime=np.mean(successtimeandcost[0])
    avgsucccost=np.mean(successtimeandcost[1])
    avgtime=np.mean(timeRec)
    avgcost=np.mean(costs)
    h=','.join([f'%0.6f' % c for c in costs ])
    iterations=','.join([f'%d' % c for c in iterations ])
    timehistory=','.join([f'%0.2f' % c for c in timeRec ])
    row=f'%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%s,%s,%s\n' %(points,average_neighbors, str(projections),projectionCount, avgtime,avgcost,avgsucctime, avgsucccost, successcount[0],successcount[1], h, iterations,timehistory) 
    write_row(expname,row)


def neighbor_exp(points,max_neighbors, expname, projections):
    write_header(expname)
    global max_iter 
    global min_cost 
    global totalRun 

    for ne in range(0, int(np.log2(max_neighbors))):
        average_neighbors=2**ne
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        successtimeandcost=[]
        successtimeandcost.append([])
        successtimeandcost.append([])
        for i in range(0,10):
            D=get_D_3projections(points)
            mv = mview.basic(D, Q=projections, average_neighbors=average_neighbors, max_iter=max_iter, min_cost=min_cost, estimate=False )
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
           
            if abs(max_iter - mv.H['iterations']) < 2:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
                successtimeandcost[0].append(mv.time)
                successtimeandcost[1].append(mv.cost)

        process_runs(expname,points,average_neighbors, projections, successcount, timeRec,costs,iterations, successtimeandcost, len(mv.Q))



 
def points_exp(max_points, expname,  average_neighbors,projection  ):
    write_header(expname)
    global max_iter 
    global min_cost 
    global totalRun 
    
    step=100
    T=int(max_points/step)+1
    for n in range(1, T):
        points=n * step
        costs=[]
        cost_success=[]
        timeRec=[]
        iterations=[]
        successcount=[0,0]
        successtimeandcost=[]
        successtimeandcost.append([])
        successtimeandcost.append([])
        D=get_D_3projections(points)
        for i in range(0,10):
            
            mv = mview.basic(D, Q=projection, average_neighbors=average_neighbors, max_iter=max_iter, min_cost=min_cost)
            costs.append(mv.cost)
            timeRec.append( mv.time)
            iterations.append(mv.H['iterations'])
            if abs(max_iter - mv.H['iterations']) < 2:
                successcount[1]=successcount[1]+1
            else:
                successcount[0]=successcount[0]+1
                successtimeandcost[0].append(mv.time)
                successtimeandcost[1].append(mv.cost)

        process_runs(expname,points,average_neighbors, projection, successcount, timeRec,costs,iterations, successtimeandcost, len(mv.Q))
 

# points_exp(2000,'1a',1,'standard') 
# points_exp(2000,'1b',1,None) 
project_exp(200,21,'2a',1,'cylinder') 
#project_exp(200,20,'2b',1,None) 
# neighbor_exp(200,200,'3a','standard')
# neighbor_exp(200,200,'3b',None)
