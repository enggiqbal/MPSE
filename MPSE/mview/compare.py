import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import misc, distances, gd, perspective, mds, tsne, gd, multiview, mpse

def all(D, persp, method='mds', separate=True, fixed=True, varying=True, verbose=0, title='', names=None, labels=None, edges=None, **kwargs):
    """
    Runs all algorithms on specified data and returns all instances.
    """
    K = len(D); N = len(D[0])
    # Separate visualization
    if separate is True:
        svis = []
        for k in range(K):
            svis.append(mds.MDS(D[k]));
            svis[k].initialize()
            svis[k].optimize(verbose=verbose,**kwargs)
    if fixed is True:
        fvis = multiview.Multiview(D,persp,verbose=verbose,labels=labels)
        fvis.setup_visualization()
        fvis.initialize_X()
        fvis.optimize_X()
    if varying is True:
        vvis = multiview.Multiview(D,persp,verbose=verbose,labels=labels)
        vvis.setup_visualization()
        vvis.initialize_X()
        vvis.initialize_Q()
        vvis.optimize_all(batch_number=10,max_iters=300)

    return svis, fvis, vvis

def plot(svis,fvis,vvis, separate=True, fixed=True, varying=True, verbose=0,
         title='', names=None, labels=None, edges=None, colors=None,**kwargs):
    """
    Runs all algorithms on specified data.
    """
    K = len(svis)
    if names is None:
        names = list(range(1,K+1))

    fig, axes = plt.subplots(3,K)
    [ax.set_axis_off() for ax in axes.ravel()]
    plt.tight_layout()
                
    # Separate visualization
    if separate is True:
        for k in range(K):
            svis[k].figureX(ax=axes[0,k],edges=edges[k],colors=colors[k])
            #axes[0,k].title.set_text(names[k]+f'\n {svis[k].ncost:0.2e}')
            axes[0,k].title.set_text(f'\n {svis[k].ncost:0.2e}')
    if fixed is True:
        fvis.figureY(axes=axes[1],edges=edges,colors=colors)
        for k in range(K):
            axes[1,k].title.set_text(f'{fvis.individual_ncost[k]:0.2e}')
        #fvis.figure(plot=False,colors=colors)
        fvis.figureX(edges=edges[0],colors=colors[0],title='marriage+loan+business / fixed',save='mlf')
    if varying is True:
        vvis.figureY(axes=axes[2],edges=edges,colors=colors)
        for k in range(K):
            axes[2,k].title.set_text(f'{vvis.individual_ncost[k]:0.2e}')
        #vvis.figure(colors=colors)
        vvis.figureX(edges=edges[0],colors=colors[0],title='marriage+loan+business / varying',save='mlv')
    
    axes[0,0].set_ylabel('individual')
    axes[1,0].set_ylabel('fixed perspectives')
    for ax in axes.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
                                  
    plt.show()

def main(D, persp, method='mds', separate=True, fixed=True, varying=True, verbose=0, title='', names=None, labels=None, edges=None, **kwargs):
    """
    Runs all algorithms on specified data.
    """
    K = len(D)
    if names is None:
        names = list(range(1,K+1))

    fig, axes = plt.subplots(3,K)
    [ax.set_axis_off() for ax in axes.ravel()]
    plt.tight_layout()
                
    # Separate visualization
    if separate is True:
        for k in range(K):
            vis = mds.MDS(D[k],labels=labels)
            vis.initialize()
            vis.optimize(verbose=verbose,**kwargs)
            vis.figureX(ax=axes[0,k],edges=edges[k])
            axes[0,k].title.set_text(names[k]+f'\n {vis.ncost:0.2e}')  
        vis.figure(plot=False)
    if fixed is True:
        vis = multiview.Multiview(D,persp,verbose=verbose,labels=labels)
        vis.setup_visualization()
        vis.initialize_X()
        vis.optimize_X()
        vis.figureY(axes=axes[1],edges=edges)
        for k in range(K):
            axes[1,k].title.set_text(f'{vis.individual_ncost[k]:0.2e}')
        vis.figure(plot=False)
        vis.figureX(edges=edges[0])
    if varying is True:
        vis = multiview.Multiview(D,persp,verbose=verbose,labels=labels)
        vis.setup_visualization()
        vis.initialize_X()
        vis.initialize_Q()
        vis.optimize_all()
        print(vis.ncost)
        vis.figureY(axes=axes[2],edges=edges)
        for k in range(K):
            axes[2,k].title.set_text(f'{vis.individual_ncost[k]:0.2e}')
        vis.figure()
        vis.figureX(edges=edges[0])
    
    axes[0,0].set_ylabel('individual')
    axes[1,0].set_ylabel('fixed perspectives')
    for ax in axes.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
                                  
    plt.show()
