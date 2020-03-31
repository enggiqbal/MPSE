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

### Older ###

def example_binomial(N=100,K=2):
    for p in [0.05,0.1,0.5,1.0]:
        D = multigraph.binomial(N,p,K=K)
        mv = MPSE(D,verbose=1)
        mv.gd(plot=True,verbose=1)
        mv.figureX()
        mv.figureHY(edges=True)
    plt.show()
    
def noisy(N=100):
    noise_levels = [0.0001,0.001,0.01,0.1,0.5]
    stress = []
    X = misc.disk(N,dim=3)
    proj = perspective.Proj()
    proj.set_params_list(special='standard')
    Y = proj.project(X)
    D = distances.compute(Y)
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        stress_best = []
        for i in range(3):
            mv = Multiview(D_noisy,persp=proj,verbose=1)
            mv.setup_visualization(visualization='mds')
            mv.initialize_X()
            mv.optimize_X(algorithm='agd')
            stress_best.append(mv.normalized_cost)
        stress.append(min(stress_best))
    fig = plt.figure()
    plt.loglog(noise_levels,stress,linestyle='--',marker='o')
    plt.title('Normalized total stress')
    plt.xlabel('noise level')
    plt.ylabel('total stress')
    plt.show()

def noise_all(N=100):
    noise_levels = [0.001,0.01,0.07,0.15,0.4]
    stress = []
    X = misc.disk(N,dim=3)
    proj = perspective.Proj(d1=2,d2=2)
    proj.set_params_list(special='identity',number=3)
    Y = proj.project(X)
    D = distances.compute(Y)
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        mv = Multiview(D_noisy,persp=proj)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(verbose=1)
        mv.optimize_X(algorithm='gd',learning_rate=1,max_iters=300,
                      verbose=1)
        stress.append(mv.cost)
    fig = plt.figure()
    plt.semilogx(noise_levels,stress)
    plt.show()

def example_random_graph_perspectives(N=100):
    probs = [0.04,0.05,0.1,0.2,0.5,1.0]
    nums = [4,5,10,20,50,100]
    Ks = [1,2,3,4,5]
    error = np.empty((len(Ks),len(probs)))
    fig = plt.figure()
    for i in range(len(probs)):
        p = probs[i]
        for j in range(len(Ks)):
            K = Ks[j]
            D = multigraph.binomial(N,p,K=K)
            if K==1: D= [D]
            persp = perspective.Persp()
            persp.fix_Q(number=K)
            vis = MPSE(D,persp=persp)
            vis.setup_visualization()
            vis.initialize_X()
            vis.initialize_Q()
            vis.optimize_all(min_step=1e-8)
            error[j,i] = max(vis.cost,1e-6)
    for i in range(len(Ks)):
        plt.semilogy(error[i],label=f'K {Ks[i]}')
    plt.ylabel('MDS stress')
    plt.xlabel('average neighbors')
    plt.xticks(range(len(nums)),nums)
    plt.legend()
    plt.tight_layout
    plt.show()
    
    
if __name__=='__main__':
    time()
