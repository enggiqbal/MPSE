import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import misc, multigraph, gd, projections, mds, tsne, plots

class MPSE(object):
    """\
    Class with methods for multi-perspective simultaneous embedding.
    """

    def __init__(self, dissimilarities, d1=3, d2=2, family='linear',
                 constraint='orthogonal', X=None, Q=None, X0=None, Q0=None,
                 visualization='mds', total_cost_function='rms', verbose=0,
                 title='',level=0,**kwargs):
        """\
        Initializes MPSE object.

        Parameters:

        D : list
        List containing dissimilarity graphs.

        persp : Object instance of projections.Persp class or int > 0.
        Describes set of allowed projection functions and stores list of
        projection parameters. See perspective.py. If instead of a Persp object
        a positive integer int is given, then it is assumed that d1=d2=int
        and that all projections are the identity. 
        """
        if verbose > 0:
            print('mpse.MPSE('+title+'):')
        self.verbose = verbose; self.title = title; self.level = level

        self.DD = multigraph.multigraph_setup(dissimilarities,Q=Q,
                                              verbose=verbose,
                                              **kwargs)
        self.D = self.DD.D

        self.K = self.DD.attributes
        self.N = self.DD.node_number

        self.d1 = d1; self.d2 = d2
        self.family = family; self.constraint=constraint
        proj = projections.PROJ(d1,d2,family,constraint)
        self.proj = proj

        if Q is True:
            assert hasattr(self.DD,'Q')
            Q = self.DD.Q
        elif isinstance(Q,str):
            Q = proj.generate(number=self.K,method=Q)
        if isinstance(Q0,str):
            Q0 = proj.generate(number=self.K,method=Q0)
            
        assert X is None or Q is None
        self.X = X; self.X_is_fixed = X is not None
        self.Q = Q; self.Q_is_fixed = Q is not None

        self.setup_visualization(visualization=visualization,**kwargs)

        if self.X_is_fixed == True:
            X0 = X
        if self.Q_is_fixed == True:
            Q0 = Q

        self.initial_cost = None
        self.initial_individual_cost = None
        self.initialize(X0=X0,Q0=Q0)
    
        if verbose > 0:
            print('  dissimilarity stats:')
            print(f'    number of views : {self.K}')
            print(f'    number of points : {self.N}')
            print('  embedding stats:')
            print(f'    embedding dimension : {self.proj.d1}')
            print(f'    projection dimension : {self.proj.d2}')

    def setup_visualization(self,visualization='mds',**kwargs):
        assert visualization in ['mds','tsne']
        self.visualization_method = visualization

        if visualization is 'mds':
            visualization_class = mds.MDS
        elif visualization is 'tsne':
            visualization_class = tsne.TSNE

        self.visualization= []
        for k in range(self.K):
            self.visualization.\
                append(visualization_class(self.D[k],self.proj.d2,
                                           verbose=0, #self.verbose,
                                           level=self.level+1,
                                           title=f'perspective # {k+1}',
                                           **kwargs))
        
        def cost_function(X,Q,Y=None):
            if Y is None:
                Y = self.proj.project(X,Q=Q)
            cost = 0
            for k in range(self.K):
                cost += self.visualization[k].cost_function(Y[k],**kwargs)**2
            return math.sqrt(cost/self.K)
        self.cost_function = cost_function

        def cost_function_all(X,Q,Y=None):
            if Y is None:
                Y = self.proj.project(Q,X)
            cost = 0; individual_cost = np.zeros(self.K)
            for k in range(self.K):
                cost_k = self.visualization[k].f(Y[k],**kwargs)
                cost += cost_k**2
                individual_cost[k] = cost_k
            cost = math.sqrt(cost/self.K)
            return cost, individual_cost
        self.cost_function_all = cost_function_all
        
        self.cost_function = cost_function
        
        if self.proj.family == 'linear':

            def F(X,Q,D=None,**kwargs):
                """\
                Returns cost and gradient at (X,Q) for dissimilarities D.
                """
                if D is None:
                    D = self.D
                cost = 0
                dX = np.zeros((self.N,self.proj.d1))
                dQ = []
                Y = self.proj.project(Q,X)
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],D=D[k])
                    cost += costk**2
                    dX += dYk @ Q[k]
                    dQ.append(dYk.T @ X)
                cost = math.sqrt(cost/self.K)
                return (cost,[dX,np.array(dQ)])
            self.F = F

            def FX(X,Q,D=None,**kwargs):
                if D is None:
                    D = self.D
                cost = 0
                dX = np.zeros((self.N,self.proj.d1))
                Y = self.proj.project(Q,X)
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],D=D[k])
                    cost += costk**2
                    dX += dYk @ Q[k]
                cost = math.sqrt(cost/self.K)
                return (cost,dX)
            self.FX = FX

            def FQ(X,Q,D=None,**kwargs):
                if D is None:
                    D = self.D
                cost = 0
                dQ = []
                Y = self.proj.project(Q,X)
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],D=D[k])
                    cost += costk**2
                    dQ.append(dYk.T @ X)
                cost = math.sqrt(cost/self.K)
                return (cost,np.array(dQ))
            self.FQ = FQ
            
        else:

            def gradient_X(X,Q,Y=None):
                pgradient = self.proj.compute_gradient(X[0],params_list=Q)
                print(pgradient[0])
                if Y is None:
                    Y = self.proj.project(X,params_list=Q)
                gradient = np.zeros((self.N,self.d1))
                for k in range(self.K):
                    gradient += self.visualization[k].gradient_function(Y[k]) \
                                @ pgradient[k]
                return gradient
            self.gradient_X = gradient_X

    def initialize(self,X0=None,Q0=None,**kwargs):
        if self.verbose > 0:
            print('  MPSE.initialize():')

        self.time = 0
        
        if X0 is None:
            if self.verbose > 0:
                print('    X0 : random')
            self.X0 = misc.initial_embedding(self.N,dim=self.proj.d1,
                                            radius=1)
        else:
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.N,self.proj.d1)
            if self.verbose > 0:
                print('    X0 : given')
            self.X0 = X0
        self.X = self.X0.copy()
        
        if Q0 is None:
            if self.verbose > 0:
                print('    Q0 : random')
            self.Q0 = self.proj.generate(number=self.K,**kwargs)
        else:
            if self.verbose > 0:
                print('    Q0 : given')
            self.Q0 = Q0
        self.Q = self.Q0.copy()         

        self.update(**kwargs)

    def smart_initialize(self,verbose=0,**kwargs):
        """\
        Computes an mds embedding (dimension d1) of the combined distances. Only
        works when self.Q_is_fixed is False (as this is unnecessary otherwhise).

        Parameters :

        X0 : None or array
        Optional initial embedding (used to compute mds embedding)
        
        Q0 : None or list of arrays
        Optional initial projection parameters.
        """
        if self.Q_is_fixed is True:
            return
        
        else:
            if self.verbose > 0:
                print('  MPSE.smart_initialize():')
                
            if self.X_is_fixed is False:
                self.DD.combine_attributes()
                D = self.DD.D0
                vis = mds.MDS(D,dim=self.proj.d1,min_grad=1e-4,
                              verbose=self.verbose)
                vis.initialize(X0=self.X0)
                vis.gd(average_neighbors=32,max_iter=30,verbose=verbose)
                self.X = vis.X
                self.update_history(H=vis.H,Q_is_fixed=True)
                
            F = lambda Q, xi=self.D: self.FQ(self.X,Q,D=xi,**kwargs)
            Xi = self.subsample_generator(average_neighbors=32)
            Q0 = np.array(self.Q)
            self.Q, H = gd.single(Q0,F,Xi=Xi,p=self.proj.restrict,max_iter=20,
                                  min_step=1e-4,verbose=verbose)
            self.update_history(H=H,X_is_fixed=True)
            return

    def update(self,H=None,**kwargs):
        if self.X is not None and self.Q is not None:
            self.Y = self.proj.project(self.Q,self.X)

            self.cost, self.individual_cost = \
                self.cost_function_all(self.X,self.Q,Y=self.Y,**kwargs)

            if self.initial_cost is None:
                self.initial_cost = self.cost
                self.initial_individual_cost = self.individual_cost

        #if H is not None:
        #    if 'cost' in self.H:
         #       H['cost'] = np.concatenate((self.H['cost'],H['cost']))
          #      H['steps'] = np.concatenate((self.H['steps'],H['steps']))
           #     H['iterations'] = self.H['iterations']+H['iterations']
            #self.H = H

    def update_history(self,H=None,X_is_fixed=None,Q_is_fixed=None,**kwargs):
        
        if hasattr(self,'H') is False:
            self.H = {}
            self.H['iterations'] = 0
            self.H['markers'] = []
            self.H['costs'] = np.array([])
            if self.X_is_fixed is False:
                self.H['X_iters'] = np.array([])
                self.H['X_steps'] = np.array([])
                self.H['X_grads'] = np.array([])
                self.H['X_lrs'] = np.array([])
            if self.Q_is_fixed is False:
                self.H['Q_iters'] = np.array([])
                self.H['Q_steps'] = np.array([])
                self.H['Q_grads'] = np.array([])
                self.H['Q_lrs'] = np.array([])

        if X_is_fixed is None:
            X_is_fixed = self.X_is_fixed
        if Q_is_fixed is None:
            Q_is_fixed = self.Q_is_fixed
            

        if H is not None:
            self.time += H['time']
            T1 = self.H['iterations']; T2 = H['iterations']
            self.H['iterations'] = T1+T2
            self.H['markers'].append(T1)
            self.H['costs'] = np.concatenate((self.H['costs'],H['costs']))
            if X_is_fixed is False and Q_is_fixed is True:
                self.H['X_iters'] = np.concatenate(
                    (self.H['X_iters'],range(T1,T1+T2)))
                self.H['X_steps'] = np.concatenate(
                    (self.H['X_steps'],H['steps']))
                self.H['X_grads'] = np.concatenate(
                    (self.H['X_grads'],H['grads']))
                self.H['X_lrs'] = np.concatenate((self.H['X_lrs'],H['lrs']))
            elif X_is_fixed is True and Q_is_fixed is False:
                self.H['Q_iters'] = np.concatenate(\
                    (self.H['Q_iters'],range(T1,T1+T2)))
                self.H['Q_steps'] = np.concatenate(
                    (self.H['Q_steps'],H['steps']))
                self.H['Q_grads'] = np.concatenate(
                    (self.H['Q_grads'],H['grads']))
                self.H['Q_lrs'] = np.concatenate(
                    (self.H['Q_lrs'],H['lrs']))
            else:
                self.H['X_iters'] = np.concatenate(
                    (self.H['X_iters'],range(T1,T1+T2)))
                self.H['X_steps'] = np.concatenate(
                    (self.H['X_steps'],H['steps'][:,0]))
                self.H['X_grads'] = np.concatenate(
                    (self.H['X_grads'],H['grads'][:,0]))
                self.H['X_lrs'] = np.concatenate(
                    (self.H['X_lrs'],H['lrs'][:,0]))
                self.H['Q_iters'] = np.concatenate(
                    (self.H['Q_iters'],range(T1,T1+T2)))
                self.H['Q_steps'] = np.concatenate(
                    (self.H['Q_steps'],H['steps'][:,1]))
                self.H['Q_grads'] = np.concatenate(
                    (self.H['Q_grads'],H['grads'][:,1]))
                self.H['Q_lrs'] = np.concatenate(
                    (self.H['Q_lrs'],H['lrs'][:,1]))
                
    def forget(self):
        self.X = self.X0; self.H = {}; self.update()

    def subsample_generator(self,edge_proportion=None,
                            average_neighbors=None,**kwargs):
        if edge_proportion is None and average_neighbors is None:
            return None
        else:
            Xi = lambda :\
                self.DD.sample(edge_proportion=edge_proportion,
                               average_neighbors=average_neighbors,
                               **kwargs)
            return Xi

    def gd(self, scheme='mm', **kwargs):
        
        if self.verbose > 0:
            print('  MPSE.gd():')
            
        if self.Q_is_fixed is True:
            if self.verbose > 0:
                print('    mpse method : fixed projections')
                print(f'    initial stress : {self.cost:0.2e}')
            F = lambda X, xi=self.D: self.FX(X,self.Q,D=xi,**kwargs)
            Xi = self.subsample_generator(**kwargs)
            self.X, H = gd.single(self.X,F,Xi=Xi,scheme=scheme,
                                  **kwargs)
            self.update_history(H=H,Q_is_fixed=True,**kwargs)

        elif self.X_is_fixed is True:
            if self.verbose > 0:
                print('    mpse method : fixed embedding')
                print(f'    initial stress : {self.cost:0.2e}')
            Xi = self.subsample_generator(**kwargs)
            F = lambda Q, xi=self.D: self.FQ(self.X,Q,D=xi,**kwargs)
            Q0 = np.array(self.Q)
            self.Q, H = gd.single(Q0,F,Xi=Xi,p=self.proj.restrict,
                                  scheme=scheme,
                                  **kwargs)
            self.update_history(H=H,X_is_fixed=True,**kwargs)

        else:
            if self.verbose > 0:
                print('    mpse method : optimize all')
                print(f'    initial stress : {self.cost:0.2e}')
            p = [None,self.proj.restrict]
            XQ = [self.X,np.array(self.Q)]
            F = lambda XQ, xi=self.D: self.F(XQ[0],XQ[1],D=xi,**kwargs)
            Xi = self.subsample_generator(**kwargs)
            XQ, H = gd.multiple(XQ,F,Xi=Xi,p=p,scheme=scheme,**kwargs)
            self.X = XQ[0]; self.Q = XQ[1];
            self.update_history(H=H,**kwargs)

        self.update()
        if self.verbose > 0:
            print(f'  Final stress : {self.cost:0.2e}')            
 
    def figureX(self,title=None,perspectives=True,
                labels=None,edges=None,colors=True,plot=True,ax=None):

        if perspectives is True:
            perspectives = []
            for k in range(self.K):
                Q = self.Q[k]
                q = np.cross(Q[0],Q[1])
                perspectives.append(q)
        else:
            perspectives = None
            
        if edges is not None:
            if isinstance(edges,numbers.Number):
                edges = edges-self.D
        if colors is True:
            colors = self.DD.node_colors
        plots.plot3D(self.X,perspectives=perspectives,edges=edges,
                     colors=colors,title=title,ax=ax)

    def figureY(self,title='projections',include_edges=False,
                include_colors=True,plot=True,
                ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,self.K)
        else:
            plot = False
        for k in range(self.K):
            if include_edges is True:
                edges_k = self.D[k]['edge_list']
            elif include_edges is False:
                edges_k = None
            else:
                edges_k = include_edges[k]
            if include_colors is True:
                colors_k = self.D[k]['node_colors'] ####
            else:
                colors_k = None
            plots.plot2D(self.Y[k],edges=edges_k,colors=colors_k,ax=ax[k],
                         **kwargs)
        plt.suptitle(title)
        if plot is True:
            plt.draw()
            plt.pause(1.0)
    
    def figureH(self,title='computations',plot=True,ax=None):
        assert hasattr(self,'H')

        if self.X_is_fixed is True or self.Q_is_fixed is True:
            windows = 2
        else:
            windows = 3
        fig, ax = plt.subplots(1,windows,figsize=(3*windows,3))
        fig.suptitle('computations')
        fig.subplots_adjust(top=0.80)
        ax[0].semilogy(self.H['costs'],linewidth=3)
        ax[0].set_title('cost')
        i = 1
        if self.X_is_fixed is False:
            ax[i].semilogy(self.H['X_iters'],self.H['X_grads'][:],
                           label='grad size', linestyle='--')
            ax[i].semilogy(self.H['X_iters'],self.H['X_lrs'][:],
                           label='lr', linestyle='--')
            ax[i].semilogy(self.H['X_iters'],self.H['X_steps'][:],
                           label='step size', linestyle='--')
            ax[i].set_title('X')
            ax[i].legend()
            ax[i].set_xlabel('iterations')
            i = 2
        if self.Q_is_fixed is False:
            ax[i].semilogy(self.H['Q_iters'],self.H['Q_grads'][:],
                           label='gradient size',linestyle='--')
            ax[i].semilogy(self.H['Q_iters'],self.H['Q_lrs'][:],
                           label='learning rate', linestyle='--')
            ax[i].semilogy(self.H['Q_iters'],self.H['Q_steps'][:],
                           label='step size',linestyle='--')
            ax[i].set_title('Q')
            ax[i].legend()
            ax[i].set_xlabel('iterations')
                
        if plot is True:
            plt.draw()
            plt.pause(0.2)
    
##### TESTS #####

def disk_fixed_perspectives(N=100,**kwargs):
    fig, ax = plt.subplots(1,5,figsize=(15,3))
    fig.suptitle('MPSE - disk data w/ fixed perspectives')
    fig.subplots_adjust(top=0.8)
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    proj = projections.PROJ(); Q = proj.generate(number=3,method='standard') ##
    DD = multigraph.DISS(N,ncolor=labels)
    for i in range(3):
        DD.add_feature(proj.project(Q[i],X))
    mv = MPSE(DD,Q=Q,verbose=2)
    ax0 = fig.add_subplot(1,5,1,projection='3d')
    mv.figureX(title='initial embedding',ax=ax0)
    mv.gd(**kwargs)
    mv.figureX(title='final embedding')
    mv.figureY()
    mv.figureH('computation history')
    plt.draw()
    plt.pause(0.2)

def disk_fixed_embedding(N=100,**kwargs):
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    proj = projections.PROJ(); Q = proj.generate(number=3,method='standard') ##
    DD = multigraph.DISS(N,ncolor=labels)
    for i in range(3):
        DD.add_feature(proj.project(Q[i],X))
    mv = MPSE(DD,X=X,verbose=1)
    mv.gd(verbose=2,**kwargs)
    mv.figureX(title='final embedding')
    mv.figureY()
    mv.figureH('computation history')
    plt.draw()
    plt.pause(0.2)

def disk(N=100,**kwargs):
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    proj = projections.PROJ(); Q = proj.generate(number=3)
    DD = multigraph.DISS(N,ncolor=labels)
    for i in range(3):
        DD.add_feature(proj.project(Q[i],X))
    mv = MPSE(DD,verbose=1,**kwargs)
    mv.figureX(title='initial embedding')
    #mv.smart_initialize(verbose=2)
    #mv.figureX(title='smart initial embedding')
    mv.gd(verbose=2,**kwargs)
    #mv.gd(verbose=2,average_neighbors=64,max_iter=15,lr=.2)
    mv.figureX(title='final embedding')
    mv.figureY()
    mv.figureH('computation history')
    plt.show()
    
if __name__=='__main__':
    print('mview.mpse : running tests')
    #disk_fixed_perspectives(100,average_neighbors=2,max_iter=100,lr=1)
    #disk_fixed_embedding(100,average_neighbors=2,max_iter=100,lr=1)
    disk(1000,average_neighbors=2,max_iter=200,estimate=True)
    plt.show()
