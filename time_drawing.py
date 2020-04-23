import pandas as pd 
import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def draw_projection_exp(filename):
    #filename='neighbor_exp_21_estimatefalse.csv'
    df=pd.read_csv(filename)
    df['avgcost2']= (df.r1c + df.r2c + df.r3c + df.r4c + df.r5c +  df.r6c + df.r7c + df.r8c + df.r9c +df.r10c )/ 10
    plt.figure()
    plt.plot(df.projection,df.avgcost2)
    #plt.plot(df.avgtime)
    plt.xlabel('projection')
    plt.ylabel('avgcosts')
    plt.savefig('expdata/drawings/'+filename+'.png')


def draw_neighbors_exp(filename):
    #filename='neighbor_exp_21_estimatefalse.csv'
    df=pd.read_csv(filename)
    plt.figure()
    plt.plot(df.neighbors,df.avgcost)
    plt.xlabel('neighbors')
    plt.ylabel('avgcosts')
    plt.savefig('expdata/drawings/'+filename+'.png')
def fixVsVar_time(f1,f2,filename):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
    plt.figure()
    plt.plot(df1.datapoints,df1.avgtime, label="variable")
    plt.plot(df2.datapoints,df2.avgtime, label="fixed")
    plt.xlabel('datapoints')
    plt.ylabel('avgtime')
    plt.suptitle('Avg time for fixed and variable projection ')
    #plt.set_title('Avg time for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')

def fixVsVar_cost(f1,f2,filename):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
    plt.figure()
    plt.plot(df1.datapoints,df1.avgcost, label="variable")
    plt.plot(df2.datapoints,df2.avgcost, label="fixed")
    plt.xlabel('datapoints')
    plt.ylabel('avgtime')
    plt.suptitle('Avg costs for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')

#filename='neighbor_exp_21_estimatefalse.csv'
draw_neighbors_exp('neighbor_exp_21_estimatefalse.csv')
draw_neighbors_exp('neighbor_exp_19_estimatefalse.csv')

#filename='project_exp_19.csv'

draw_projection_exp('project_exp_19.csv')


filename='veriable_average_neighbors_19.csv'
fixVsVar_time('veriable_average_neighbors_19.csv', 'fixed_average_neighbors_19.csv','fixVsVar_agv_neighbor4_time')
fixVsVar_cost('veriable_average_neighbors_19.csv', 'fixed_average_neighbors_19.csv','fixVsVar_agv_neighbor4_cost')