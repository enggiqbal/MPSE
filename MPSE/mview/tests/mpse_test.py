import sys
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

sys.path.insert(1, '../')
import misc, multigraph, projections, mpse

def time():
    print('\n***mpse_test.time()***')
    N = [int(10**a) for a in [1,1.5,2,2.5]]
    repeats = 3
    successes = np.zeros(len(N))
    ratios = np.zeros(len(N))
    time = np.zeros(len(N))
    for i in range(len(N)):
        for j in range(repeats):
            X = misc.disk(N[i],dim=3)
            proj = projections.PROJ()
            Q = proj.generate(number=3, method='standard')
            D = multigraph.multigraph_from_projections(proj,Q,X)
            vis = mpse.MPSE(D,verbose=1)
            vis.gd(min_step=1e-4,verbose=1)
            if vis.cost < 1e-3:
                successes[i] += 1
                time[i] += vis.H['time']
        if successes[i] != 0:
            time[i] /= successes[i]
            ratios[i] = successes[i]/repeats

    fig = plt.plot()
    plt.loglog(N,time)
    plt.xlabel('number of points')
    plt.ylabel('time')
    plt.title('computation time')
    plt.show()

def reliability0(N=100,trials=3,repeats=5,**kwargs):
    """\
    Check number of times the mds algorithm is able to reach the optimal
    solution for different random embeddings.
    """
    print('\n***mds.reliability()***')
    cost = np.empty((repeats,trials))
    success = np.zeros(repeats)
    for i in range(repeats):
        X = misc.disk(N,2)
        D = multigraph.graph_from_coordinates(X,**kwargs)
        for j in range(trials):
            vis = mds.MDS(D,dim=2,**kwargs)
            vis.gd(min_step=1e-3,**kwargs)
            cost[i,j] = vis.cost
            if vis.cost < 1e-3:
                success[i] += 1
    print(cost)
    print(success)

def reliability(N=[100],edge_probability=[None],trials=10,**kwargs):
    """\
    Check number of times the mds algorithm is able to reach the optimal
    solution for different random embeddings.
    """
    print('\n***mds.reliability()***')
    lenN = len(N)
    lenE = len(edge_probability)
    success_count = np.zeros((lenN,lenE))
    time_ave = np.zeros((lenN,lenE))
    for i in range(lenN):
        n = N[i]
        for j in range(lenE):
            p = edge_probability[j]
            for k in range(trials):
                X = misc.disk(n,2)
                D = multigraph.graph_from_coordinates(X,**kwargs)
                vis = mds.MDS(D,dim=2,**kwargs)
                vis.gd(min_step=1e-4,edge_probability=p,max_iters=1e6,**kwargs)
                if vis.cost < 1e-2:
                    success_count[i,j] += 1
                    time_ave[i,j] += vis.H['time']
            if success_count[i,j] != 0:
                time_ave[i,j] /= success_count[i,j]
            print('N, prob :',n,p)
            print('  success ratio :',success_count[i,j]/trials)
            print('  average time :',time_ave[i,j])
    success_ratio = success_count/trials

    plt.figure()
    df = pd.DataFrame(time_ave.T,E,N)
    sn.set(font_scale=1.4)
    sn.heatmap(df, annot=success_ratio.T, cbar_kws={'label': 'time'})
    plt.xlabel('number of nodes')
    plt.ylabel('edge probabilities')
    plt.title('Success ratio')
    plt.show()
    
if __name__=='__main__':
    time()
