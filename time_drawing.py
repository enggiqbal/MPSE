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
    plt.plot(df1.points,df1.avgtime,  linestyle='dotted', marker='o', label="variable")
    plt.plot(df2.points, df2.avgtime,  marker='o', label="fixed")
    plt.xlabel('points')
    plt.ylabel('avg time')
    plt.suptitle('Avg time for fixed and variable projection ')
    #plt.set_title('Avg time for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')


def fixVsVar_cost(f1,f2,filename):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
 
    plt.figure()
    plt.plot(df1.points, df1.avgsucccost,  linestyle='dotted', marker='o', label="variable")
    plt.plot(df2.points, df2.avgsucccost,  marker='o', label="fixed")

    plt.xlabel('points')
    plt.ylabel('avg cost (global minima)')
    plt.suptitle('Avg global minima costs for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')


def draw(f1,f2,filename,title, x, y, lloc="lower right", factor=1):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
 
    plt.figure()
    plt.plot(df1[x],factor* df1[y], marker='x', linestyle='dotted', label="variable")
    plt.plot(df2[x],factor* df2[y],  marker='o', label="fixed")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle(title)
    plt.legend(loc=lloc)
    plt.savefig('expdata/drawings/'+filename+'.png')

def projection_time(f1,f2,filename):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
 
    plt.figure()
    plt.plot(df1.points, df1.avgsucccost,  marker='x', linestyle='dotted',   label="variable")
    plt.plot(df2.points, df2.avgsucccost,  marker='o', label="fixed")

    plt.xlabel('points')
    plt.ylabel('avg cost (global minima)')
    plt.suptitle('Avg global minima costs for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')


def fixVsVar_success(f1,f2,filename):
    df1=pd.read_csv(f1)
    df2=pd.read_csv(f2)
 
    plt.figure()
    plt.plot(df1.points,100*df1.success/10,  linestyle='dotted', marker='o', label="variable")
    plt.plot(df2.points,100*df2.success/10,  marker='o', label="fixed")

    plt.xlabel('points')
    plt.ylabel('Global minima count')
    plt.suptitle('Global minima costs for fixed and variable projection ')
    plt.legend(loc="upper left")
    plt.savefig('expdata/drawings/'+filename+'.png')

 
draw('expdata/1a_revised.csv', 'expdata/1b_revised.csv','1a_1b_points_cost_revised', 'avg cost for variable points','points','avgcost')
draw('expdata/1a_revised.csv', 'expdata/1b_revised.csv','1a_1b_points_time_revised', 'avg time for variable points','points','avgtime')
#draw('expdata/1a_revised.csv', 'expdata/1b_revised.csv','1a_1b_points_gm_revised', ' percentage of global minima for variable points','points','success', factor=10)

draw('expdata/2a_revised.csv', 'expdata/2b_revised.csv','2a_2b_projection_time_revised', 'average time for variable projection for fixed iterations','projectioncount','avgtime')
 #draw('expdata/2a_revised.csv', 'expdata/2b_revised.csv','2a_2b_projection_gm_revised', 'percentage of global minima for variable projection','projectioncount','success', factor=10)
draw('expdata/2a_revised.csv', 'expdata/2b_revised.csv','2a_2b_projection_cost_revised', 'avg cost for variable projections','projectioncount','avgcost', lloc="upper right" )
 