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

        self.D = multigraph.multigraph_setup(dissimilarities)

        self.K = self.D['attribute_number']
        self.N = self.D['node_number']

        self.d1 = d1; self.d2 = d2
        self.family = family; self.constraint='orthogonal'
        proj = projections.PROJ(d1,d2,family,constraint)

        assert X is None or Q is None
        if X is None:
            self.Xfixed = False
        else:
            self.Xfixed = True
        self.X = X
        if Q is None:
            self.Qfixed = False
            self.Q = Q
        else:
            self.Qfixed = True
            if isinstance(Q,str):
                self.Q = proj.generate(number=self.K,
                                       method=Q)
            else:
                self.Q = Q
                proj.check(Q=self.Q)
        self.proj = proj

        self.X0 = X0
        self.Q0 = Q0

        self.setup_visualization(visualization=visualization,**kwargs)

        self.H = {}
        
        if verbose > 0:
            print('  dissimilarity stats:')
            print(f'    number of views : {self.K}')
            print(f'    number of points : {self.N}')
            print(f'    dissimilarity rms : {self.D["rms"]:0.2e}')
            print(f'    normalization factor : {self.D["normalization"]:0.2e}')
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
                                           verbose=self.verbose,
                                           level=self.level+1,
                                           title=f'perspective # {k+1}',
                                           **kwargs))
        
        def cost_function(X,Q,Y=None):
            if Y is None:
                Y = self.proj.project(X,Q=Q)
            cost = 0
            for k in range(self.K):
                cost += self.visualization[k].cost_function(Y[k])**2
            return math.sqrt(cost)
        self.cost_function = cost_function

        def cost_function_k(X,q,k,y=None):
            if y is None:
                y = self.proj.project(X,q=q)
            cost_k = self.visualization[k].cost_function(y)
            return cost_k
        self.cost_function_k = cost_function_k

        def cost_function_all(X,Q,Y=None):
            if Y is None:
                Y = self.proj.project(X,Q=Q)
            cost = 0; individual_cost = np.zeros(self.K)
            for k in range(self.K):
                cost_k = self.visualization[k].f(Y[k])
                cost += cost_k
                individual_cost[k] = cost_k
            return cost, individual_cost
        self.cost_function_all = cost_function_all
        
        self.cost_function = cost_function
        
        if self.proj.family == 'linear':

            def F(X,Q,Y=None,**kwargs):
                if Y is None:
                    Y = self.proj.project(Q,X)
                cost = 0
                dX = np.zeros((self.N,self.proj.d1))
                dQ = []
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],**kwargs)
                    cost += costk**2
                    dX += dYk @ Q[k]
                    dQ.append(dYk.T @ X)
                cost = math.sqrt(cost)
                return (cost,[dX,np.array(dQ)])
            self.F = F

            def FX(X,Q,Y=None,**kwargs):
                if Y is None:
                    Y = self.proj.project(Q,X)
                cost = 0
                dX = np.zeros((self.N,self.proj.d1))
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],**kwargs)
                    cost += costk**2
                    dX += dYk @ Q[k]
                return (math.sqrt(cost),dX)
            self.FX = FX

            def FQ(X,Q,Y=None,**kwargs):
                if Y is None:
                    Y = self.proj.project(Q,X)
                cost = 0
                dQ = []
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],**kwargs)
                    cost += costk
                    dQ.append(dYk.T @ X)
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

    def initialize_Q(self, **kwargs):
        """\
        Set initial parameters for perspective functions.
        """
        if self.verbose > 0:
            print('  Multiview.initialize_Q():')
        self.Q = self.proj.generate(number=self.K,**kwargs)
        self.Q0 = self.Q.copy()
        self.update()
        
    def initialize_X(self, X0=None, method='random',max_iters=50,**kwargs):
        """\
        Set initial embedding using misc.initial function.

        Parameters:

        Y0 : numpy array
        Initial embedding (optional)

        number : int > 0
        Number of initial embeddings to be generated and saved. When looking for
        a minimizer of the stress function, the optimization algorithm is run
        using the different initial embeddings and the best solution is 
        retained.
        """
        if self.verbose > 0:
            print('  MPSE.initialize_X():')

        if X0 is not None:
            if self.verbose > 0:
                print('    method : X0 given')
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.N,self.proj.d1)
            self.X = X0
        else:
            if self.verbose > 0:
                print('    method : ',method)
            if method == 'random':
                self.X = misc.initial_embedding(self.N,dim=self.proj.d1,
                                                radius=1)
                                                #radius=self.D_rms,**kwargs)
            elif method == 'mds':
                D = multigraph.combine(self.D) #= np.average(self.D,axis=0)
                vis = mds.MDS(D,dim=self.proj.d1)
                vis.initialize()
                vis.gd(max_iters=max_iters,method='mm',verbose=2,
                       plot=True,**kwargs)
                self.X = vis.X
        self.update()
        self.X0 = self.X.copy()

    def update(self,H=None):
        if self.X is not None and self.Q is not None:
            self.Y = self.proj.project(self.Q,self.X)

            self.cost, self.individual_cost = self.cost_function_all(self.X,self.Q,Y=self.Y)
            #self.individual_ncost = np.sqrt(self.individual_cost/(self.N*(self.N-1)/2))/self.individual_D_rms
        if H is not None:
            if bool(self.H) is True:
                H['cost'] = np.concatenate((self.H['cost'],H['cost']))
                H['steps'] = np.concatenate((self.H['steps'],H['steps']))
                H['iterations'] = self.H['iterations']+H['iterations']
            self.H = H  

    def forget(self):
        self.X = self.X0; self.H = {}; self.update()

    def gd(self, step_rule='mm', min_step=1e-6,**kwargs):
        if self.Q is None:
            self.initialize_Q(title='automatic')
        if self.X is None:
            self.initialize_X(title='automatic')

        if self.verbose > 0:
            print('  MPSE.gd():')
            
        if self.Qfixed is True:
            if self.verbose > 0:
                print('    mpse method : fixed projections')
                print(f'    initial stress : {self.cost:0.2e}')
            F = lambda X : self.FX(X,self.Q)
            self.X, H = gd.single(self.X,F,step_rule=step_rule,
                                  min_step=min_step,**kwargs)
            self.update(H=H)

        elif self.Xfixed is True:
            if self.verbose > 0:
                print('    mpse method : fixed embedding')
                print(f'    initial stress : {self.cost:0.2e}')
            F = lambda Q: self.FQ(self.X,Q,**kwargs)
            Q0 = np.array(self.Q)
            self.Q, H = gd.single(Q0,F,p=self.proj.restrict,
                                  step_rule=step_rule,min_step=min_step,
                                  **kwargs)
            self.update(H=H)

        else:
            if self.verbose > 0:
                print('    mpse method : optimize all')
                print(f'    initial stress : {self.cost:0.2e}')
            p = [None,self.proj.restrict]
            XQ = [self.X,np.array(self.Q)]
            F = lambda XQ: self.F(XQ[0],XQ[1],**kwargs)
            XQ, H = gd.multiple(XQ,F,p,step_rule=step_rule,
                                min_step=min_step,**kwargs,**self.H)
            self.X = XQ[0]; self.Q = XQ[1]; self.update(H=H)
            
        if self.verbose > 0:
            print(f'  Final stress : {self.cost:0.2e}')            
 
    def figureX(self,title='Final embedding',perspectives=True,
                labels=None,edges=None,colors=None,plot=True,save=False):

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
        plots.plot3D(self.X,perspectives=perspectives,edges=edges,
                     colors=colors,title=title,save=save)

    def figureY(self,title='perspectives',edges=False,colors=True,plot=True,
                ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,self.K)
        else:
            plot = False
        for k in range(self.K):
            if edges is True:
                edges_k = self.D[k]['edges']
            elif edges is False:
                edges_k = None
            else:
                edges_k = edges[k]
            if colors is True:
                colors_k = self.D[k]['colors']
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
        if ax is None:
            fig, ax = plt.subplots()
        plots.plot_cost(self.H['costs'],self.H['steps'],title=title,ax=ax)
        if plot is True:
            plt.draw()
            plt.pause(0.2)

    def figureHY(self,title='multiview computation & embedding',colors=True,
               edges=False,plot=True):
        assert self.proj.d2 >= 2
        fig,axs = plt.subplots(1,self.K+1)
        plt.suptitle(title+f', cost : {self.cost:0.2e}')
        self.figureH(ax=axs[0])
        self.figureY(ax=axs[1::],colors=colors,edges=edges)
        if plot is True:
            plt.draw()
            plt.pause(1)
    
##### TESTS #####

def disk(N=100,Qfixed=False,Xfixed=False,**kwargs):
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    proj = projections.PROJ(); Q = proj.generate(number=3,method='standard')
    D = multigraph.multigraph_from_projections(proj,Q,X,**kwargs)
    if Qfixed is True:
        mv = MPSE(D,Q=Q,verbose=1)
    elif Xfixed is True:
        mv = MPSE(D,X=X,verbose=1)
    else:
        mv = MPSE(D,verbose=1)
    mv.gd(verbose=2,plot=True,**kwargs)
    mv.figureX()
    mv.figureHY()
    plt.show()

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
    
### Quick plots ###

def xyz():
    X = np.load('raw/xyz.npy')
    persp = perspective.Persp()
    persp.fix_Q(number=3,special='standard')
    D = multigraph.binomial(N=1000,p=.01,K=3)
    mv = MPSE(D,persp=persp,verbose=1)
    mv.setup_visualization()
    mv.initialize_X(X)
    mv.figureY(plot=True,title='',axis=False)
    plt.show()

    
if __name__=='__main__':
    disk(30,Qfixed=True,edge_max_distance=2,edge_probability=.5,max_iter=300)
    #disk(30,Xfixed=True)
    #disk(30)
    #example_binomial(N=30,K=3)
    #noisy()
    #noisy_combine()
    #test_mds0()
    #test_mds123()#save_data=True)
    #example_random_graph_perspectives(N=100)
    #xyz()

