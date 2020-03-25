import sys
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, '../')
import misc, multigraph, mds

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

    fig, ax = plt.subplots()
    im = ax.matshow(success_ratio.T)
    ax.set_xticklabels(N)
    ax.set_yticklabels(E)
    for i in range(lenE):
        for j in range(lenN):
            if success_count[j,i] != 0:
                string = f'{success_ratio[j,i]:0.2f}\n{time_ave[j,i]:0.2e}'
                text = ax.text(j, i, string,
                               ha="center", va="center", color="w")
    ax.set_title("Success rato and time")
    fig.tight_layout()
    plt.show()
        
if __name__=='__main__':
    N = [int(10**a) for a in [1,1.5]]
    E = [1.0,.8,.6,.4,.2]
    reliability(N,E,trials=5)
