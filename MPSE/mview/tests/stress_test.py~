import sys
import numbers, math, random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from scipy.spatial import distance_matrix

sys.path.insert(1, '../')
import misc, multigraph, mds

### Tests to quantify what a good normalized stress score is ###

def stress_vs_noise(N=128,dim=2):
    X = misc.disk(N,dim=dim)
    D = distance_matrix(X,X)
    vis = mds.MDS(D,dim=dim)

    noise = 10**np.arange(-2,3,0.5)
    its = len(noise)
    stress = np.empty(its)
    for i in range(its):
        X_noisy = X+np.random.randn(N,dim)*noise[i]
        stress[i] = mds.stress_function(X_noisy,D,estimate=False)
        vis.initialize(X0=X_noisy)
        vis.figureX(title=f'noise: {noise[i]:0.2e}, stress: {stress[i]:0.2e}')
    plt.show()
    return

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
    df = pd.DataFrame(success_ratio.T,E,N)
    sn.set(font_scale=1.4)
    sn.heatmap(df, annot=time_ave.T, cbar_kws={'label': 'success ratio'})
    plt.xlabel('number of nodes')
    plt.ylabel('edge probabilities')
    plt.title('computation time')
    plt.show()

    
def test1(N=100,trials=3,repeats=5,**kwargs):
    print('\n***mds.test1()***')
    cost = np.empty((repeats,trials))
    for i in range(repeats):
        X = misc.disk(N,2)
        D = multigraph.graph_from_coordinates(X,**kwargs)
        for j in range(trials):
            mds = MDS(D,dim=2,**kwargs)
            mds.gd(min_step=1e-3,**kwargs)
            cost[i,j] = mds.cost
    print(cost)

def test_gd_lr(N=100,dim=2):
    print('\n***mds.gd_lr()***')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y) 
    D = multigraph.from_coordinates(Y,colors=colors)
    title = 'recovering random coordinates for different learning rates'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    for lr in [100,10,1,.1]:
        mds.gd(lr=lr)
        mds.figure(title=f'lr = {lr}')
        mds.forget()
    plt.show()

def example_stochastic(N=100,dim=2):
    print('\n***mds.example_stochastic()***\n')
    
    Y = misc.disk(N,dim); colors = misc.labels(Y)
    
    D = multigraph.from_coordinates(Y,colors=colors)

    title = 'recovering random coordinates from full dissimilarity matrix ' +\
            'using SGD, same learning rate, and different approx'
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize()
    for approx in [1.,.8,.6,.4,.2,.1]:
        mds.stochastic(verbose=1,lr=10.0,min_step=1e-6,
                       approx=approx,title=f'SGD using {approx} of edges')
        mds.figure(title=f'approx = {approx}, time = {mds.H["time"]:0.2f}')
        mds.forget()
    plt.show()
    
def example_weights(N=100,dim=2):
    print('\n***mds.example_weights()***\n')
    print('Here we explore the MDS embedding for a full graph for different'+
          'weights')
    title='MDS embedding for multiple weights'
    X = misc.disk(N,dim); colors = misc.labels(X)
    X0 = misc.disk(N,dim)
    
    D = multigraph.from_coordinates(X,colors=colors)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'absolute weights')

    multigraph.set_weights(D,scaling=.5)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'1/sqrt(Dij) weights')

    multigraph.set_weights(D,scaling=1)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'1/Dij weights')

    multigraph.set_weights(D,scaling=2)
    mds = MDS(D,dim=dim,verbose=1,title=title)
    mds.initialize(X0=X0)
    mds.stochastic(verbose=1,max_iters=50,approx=.6,lr=50)
    mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
    mds.figure(title=f'relative weights')

    plt.show()
    
def example_fewer_edges(N=100,dim=2):
    print('\n***mds.example_fewer_edges()***\n')
    print('Here we explore the MDS embedding for a full graph as far way edges'
          +'are removed')
    title='MDS embedding for multiple proportion of edges'
    X = misc.disk(N,dim); colors = misc.labels(X)
    D = multigraph.from_coordinates(X,colors=colors)
    X0 = misc.disk(N,dim)*.5
    for prop in [.99,.8,.6,.4,.2]:
        DD = multigraph.remove_edges(D,proportion=prop)
        mds = MDS(DD,dim=dim,verbose=1,title=title)
        mds.initialize(X0=X0)
        mds.stochastic(verbose=1,max_iters=300,approx=.99,lr=.5)
        mds.adaptive(verbose=1,min_step=1e-6,max_iters=300)
        mds.figure(title=f'proportion = {prop:0.1f}')
    plt.show()

def example_random_graph(N=100,dim=2):
    print('\n***mds.example_random_graph()***\n')
    print('Here we explore the MDS embedding for a random binomial graph with'+\
          'different edge probabilities.')
    fig, axes = plt.subplots(2,3)
    #[ax.set_axis_off() for ax in axes.ravel()]
    plt.tight_layout()
    for p, ax in zip([0.01,0.02,0.03,0.05,0.1,1.0],axes.ravel()):
        D = multigraph.binomial(N,p)
        mds = MDS(D,dim=dim,verbose=1)
        mds.initialize()
        mds.stochastic(max_iters=100,approx=.6,lr=.5)
        mds.agd(min_step=1e-6)
        mds.figureX(ax=ax,edges=True)
        ax.set_xlabel(f'ave. neighs. : {int(100*p)}')
        ax.set_title(f'stress = {mds.cost:0.2e}')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()

def example_random_graph_2(N=100):
    print('\n***mds.example_random_graph()***\n')
    print('Here we explore the MDS embedding for a random binomial graph with'+\
          'different edge probabilities.')
    probs = [0.04,0.05,0.1,0.2,0.5,1.0]
    nums = [4,5,10,20,50,100]
    dims = [2,3,4,5,10,20]
    error = np.empty((len(dims),len(probs)))
    fig = plt.figure()
    for i in range(len(probs)):
        p = probs[i]
        D = multigraph.binomial(N,p)
        for j in range(len(dims)):
            dim = dims[j]
            mds = MDS(D,dim=dim)
            mds.initialize()
            mds.stochastic(max_iters=100,approx=.3,lr=5)
            mds.stochastic(max_iters=100,approx=.6,lr=10)
            mds.stochastic(max_iters=100,approx=.9,lr=15)
            mds.agd(min_step=1e-8)
            error[j,i] = max(mds.cost,1e-6)
    for i in range(len(dims)):
        plt.semilogy(error[i],label=f'dim {dims[i]}')
    plt.ylabel('MDS stress')
    plt.xlabel('average neighbors')
    plt.xticks(range(len(nums)),nums)
    plt.legend()
    plt.tight_layout
    plt.show()
    
def disk_compare(N=100,dim=2): ###
    print('\n***mds.disk_compare()***')
    
    X = misc.disk(N,2); labels = misc.labels(X)
    
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.title('original data')
    plt.draw()
    plt.pause(0.1)
    
    D = distances.compute(X)
    
    mds = MDS(D,dim=dim,verbose=1,title='disk experiments',labels=labels)
    mds.initialize()
    mds.figureX(title='initial embedding')

    title = 'full gradient & agd'
    mds.optimize(algorithm='agd',verbose=2,label=title)
    mds.figureX(title=title)
    mds.figureH(title=title)

    mds.forget()
    title = 'approx gradient & gd'
    mds.approximate(algorithm='gd',verbose=2,label=title)
    mds.figureX(title=title)
    mds.figureH(title=title)

    mds.forget()
    title = 'combine'
    mds.approximate(algorithm='gd',verbose=2,label=title)
    mds.optimize(verbose=2,label=title,max_iters=10)
    mds.figureX(title=title)
    mds.figureH(title=title)
    plt.show()

def example_disk_noisy(N=100,dim=2):
    print('\n***mds.example_disk_noisy()***\n')
    noise_levels = [0.001,0.005,0.01,0.03,0.07,0.1,0.15,0.2,0.7,1.0]
    stress = []
    Y = misc.disk(N,dim)
    D = distances.compute(Y)
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        mds = MDS(D_noisy,dim,verbose=1,title=f'noise : {noise:0.2f}')
        mds.initialize()
        mds.optimize(algorithm='agd',max_iters=300,verbose=1)
        stress.append(mds.ncost)
    fig = plt.figure()
    plt.loglog(noise_levels,stress,'.-')
    plt.xlabel('noise level')
    plt.ylabel('stress')
    plt.title('Normalized MDS stress for various noise levels')
    plt.show()
    
def example_disk_dimensions(N=100):
    print('\n***mds.example_disk_dimensions()***\n')
    dims = range(1,11)
    stress = []
    for dim in dims:
        Y = misc.disk(N,dim)
        D = distances.compute(Y)
        mds = MDS(D,dim,verbose=1,label=f'dimension : {dim}')
        mds.initialize_Y()
        mds.optimize(algorithm='agd',max_iters=300)
        stress.append(mds.ncost)
    fig = plt.figure()
    plt.semilogy(dims,stress)
    plt.xlabel('dimension')
    plt.ylabel('stress')
    plt.title('Normalized MDS stress for various dimensions')
    plt.show()

### EMBEDDABILITY TESTS ###

def embeddability_dims(ax=None):
    print('\n**mds.embeddability_dims()')
    N=50
    ncost = []
    dims = list(range(2,50,5))
    #XX = misc.disk(N,20)
    XX = misc.box(N,20)
    for dim in dims:
        X = XX[:,0:dim]
        D = multigraph.coord2dict(X,weights='relative')
        mds = MDS(D,dim=2,verbose=1)
        mds.initialize()
        mds.optimize()
        ncost.append(mds.ncost)
    if ax is None:
        fig, ax = plt.subplots(1)
        plot = True
    else:
        plot = False
    ax.plot(dims,ncost)
    if plot is True:
        plt.show()

def embeddability_noise(ax=None):
    print('\n**mds.embeddability_noise()')
    N=50
    ncost = []
    noise_list = [0]+10**np.arange(-2,1,0.5)
    X = misc.disk(N,4)
    DD = distances.compute(X)
    for noise in noise_list:
        D = DD*(1+np.random.randn(N,N)*noise)
        mds = MDS(D,dim=4,verbose=1)
        mds.initialize()
        mds.optimize()
        ncost.append(mds.ncost)
    if ax is None:
        fig, ax = plt.subplots(1)
        plot = True
    else:
        plot = False
    ax.semilogx(noise_list,ncost)
    if plot is True:
        plt.show()
        
if __name__=='__main__':

    ### Quantification of normalized stress ###
    stress_vs_noise(N=1024,dim=2)
    
    #N = [int(10**a) for a in [1,1.5,2,2.5,3]]
    #E = [1.0,.9,.8,.7]
    #reliability(N,E,trials=10)
