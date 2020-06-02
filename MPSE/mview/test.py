import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.spatial import distance_matrix

import misc, distances, perspective, mds, multiview
from multiview import Multiview

### Tests to check whether adaptive gradient descent converges to optimal solution of MDS or multiview-MDS problems, depending on distance matrix and initial parameters.

def agd_mds_standard(N=100,dim=2,trials=3,runs=5):
    """\
    test convergence of adaptive gradient descent for MDS problem, by solving 
    the MDS problems with multiple initial parameters.
    """
    print('*** test.agd_mds_standard() ***')
    print(f'  N : {N}')
    print(f'  dim : {dim}')
    print()

    for i in range(trials):
        print(f'  Trial # {i}')
        print('  Normalized cost :')
        Y = misc.disk(N,dim)
        D = distances.compute(Y)
        vis = mds.MDS(D,dim=dim)
        stress = []
        for run in range(runs):
            vis.initialize_Y()
            vis.optimize(algorithm='agd')
            stress.append(vis.ncost)
            print(f'  {vis.ncost:0.2e}')
        print()
        
def agd_multiview_mds_standard(N=100,trials=3,runs=5):
    """\
    Test convergence of adaptive gradient descent for multiview MDS with
    standard projections, by solving a single multiview MDS problem using
    multiple initial parameters.
    """
    print('*** test.agd_multiview_mds_standard() ***')
    print(f'  N : {N}')
    print()

    for i in range(trials):
        print(f'  Trial # {i}')
        print('  Normalized cost :')
        X = misc.disk(N,dim=3)
        persp = perspective.Persp()
        persp.fix_Q(special='standard')
        Y = persp.compute_Y(X)
        D = distances.compute(Y)
        mv = multiview.Multiview(D,persp=persp)
        mv.setup_visualization(visualization='mds')
        stress = []
        for run in range(runs):
            mv.initialize_X()
            mv.optimize_X(algorithm='agd')
            stress.append(mv.ncost)
            print(f'  {mv.ncost:0.2e}')
        print()

def varying_view_number(N=100,runs=1):
    """\
    Multiview-MDS experiment with varying number of views
    """
    view_number = range(1,11)
    
    print('*** test.varying_view_number() ***')
    print(f'  N : {N}')
    print()

    X = misc.disk(N,dim=3)
    persp = perspective.Persp(dimX=3,dimY=2,family='linear',
                            restriction='orthogonal')
    cost = []
    for i in view_number:
        persp.fix_Q(number=i,random='orthogonal')
        Y = persp.compute_Y(X)
        D = distances.compute(Y)
        mv = multiview.Multiview(D,persp=persp)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=runs)
        mv.optimize_X(algorithm='agd')
        cost.append(mv.ncost)
        print(f'  {i:>2} : {mv.ncost:0.2e}')
    print()
    
def compare_multiview_standard(N=100,runs=1):
    noise_levels = [0.0001,0.001,0.01,0.1,0.5]
    
    X = misc.disk(N,dim=3)
    persp = perspective.Persp(dimX=3,dimY=2)
    persp.fix_Q(special='standard',number=3)
    Y = persp.compute_Y(X)
    D = distances.compute(Y)
    
    persp2 = perspective.Persp(dimX=2,dimY=2)
    persp2.fix_Q(special='identity',number=3)
    persp3 = perspective.Persp(dimX=3,dimY=3)
    persp3.fix_Q(special='identity',number=3)

    cost = []; cost2 = []; cost3 = []; costm = []
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        
        mv = Multiview(D_noisy,persp=persp,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=1)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp2,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=runs)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost2.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp3,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=1)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost3.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_Q()
        mv.initialize_X(number=1)
        mv.optimize_all(algorithm='agd',max_iters=[20,10],rounds=20)
        costm.append(mv.ncost)
        
    fig = plt.figure()
    plt.loglog(noise_levels,cost,linestyle='--',marker='o',
               label='multi-perspective')
    plt.loglog(noise_levels,cost2,linestyle='--',marker='o', label='combine 2')
    plt.loglog(noise_levels,cost3,linestyle='--',marker='o', label='combine 3')
    plt.loglog(noise_levels,costm,linestyle='--',marker='o', label='multi-all')
    plt.legend()
    plt.xlabel('noise level')
    plt.ylabel('normalized stress')
    plt.show()

def compare_multiview_same(N=100,runs=1):
    noise_levels = [0.0001,0.001,0.01,0.1,0.5]
    stress = []
    X = misc.disk(N,dim=2)
    persp = perspective.Persp(dimX=2,dimY=2)
    #persp.fix_Q(random='orthogonal',number=3)
    persp.fix_Q(special='identity',number=3)
    Y = persp.compute_Y(X)
    D = distances.compute(Y)

    persp1 = perspective.Persp(dimX=3,dimY=2)
    persp1.fix_Q(special='standard',number=3)
    persp2 = perspective.Persp(dimX=2,dimY=2)
    persp2.fix_Q(special='identity',number=3)
    persp3 = perspective.Persp(dimX=3,dimY=3)
    persp3.fix_Q(special='identity',number=3)

    cost = []; cost2 = []; cost3 = []; costm = []
    for noise in noise_levels:
        D_noisy = distances.add_noise(D,noise)
        
        mv = Multiview(D_noisy,persp=persp1,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=1)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp2,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=runs)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost2.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp3,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_X(number=1)
        mv.optimize_X(algorithm='agd',max_iters=200)
        cost3.append(mv.ncost)

        mv = Multiview(D_noisy,persp=persp1,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_Q()
        mv.initialize_X(number=1)
        mv.optimize_all(algorithm='agd',max_iters=[30,20],rounds=40)
        costm.append(mv.ncost)
        
    fig = plt.figure()
    plt.loglog(noise_levels,cost,linestyle='--',marker='o',
               label='multi-perspective')
    plt.loglog(noise_levels,cost2,linestyle='--',marker='o', label='combine 2')
    plt.loglog(noise_levels,cost3,linestyle='--',marker='o', label='combine 3')
    plt.loglog(noise_levels,costm,linestyle='--',marker='o', label='multi-all')
    plt.legend()
    plt.xlabel('noise level')
    plt.ylabel('normalized stress')
    plt.show()
    
if __name__=='__main__':
    #agd_mds_standard(N=30,dim=2)
    #agd_multiview_mds_standard(30,trials=5,runs=5)
    #varying_view_number(N=200,runs=3)
    #compare_multiview_standard(N=30,runs=3)
    compare_multiview_same(N=10,runs=3)
