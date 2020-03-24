import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import misc, distances, multigraph, gd, perspective, mds2, tsne, plots

class MPSE(object):
    """\
    Collection of methods for multi-perspective simultaneous embedding.
    """

    def __init__(self, D, persp=2, verbose=0, title=''):
        """\
        Initializes MPSE method.

        Parameters:

        D : list
        List containing dissimilarity graphs.

        persp : Object instance of projections.Persp class or int > 0.
        Describes set of allowed projection functions and stores list of
        projection parameters. See perspective.py. If instead of a Persp object
        a positive integer int is given, then it is assumed that dimX=dimY=int
        and that all projections are the identity. 
        """
        if verbose > 0:
            print('+ mpse.MPSE('+title+'):')
        self.verbose = verbose; self.title = title

        self.D = D
        self.N = len(D[0]['nodes'])
        self.K = len(D)

        if 'colors' in D[0]:
            self.colors = D[0]['colors']
        else:
            self.colors = None
        #self.individual_D_rms = np.sqrt(np.sum(D**2,axis=(1,2))/(self.N*(self.N-1)))
        #self.D_rms = np.sqrt(np.sum(D**2)/(self.N*(self.N-1)*self.K))

        if isinstance(persp,int):
            dim = persp; assert dim  > 0
            persp = perspective.Persp(dimX=dim,dimY=dim,family='linear')
            persp.fix_Q(number=self.K,special='identity')
            self.Q = persp.Q
        else:
            if persp.fixed_Q:
                assert persp.K == self.K
                self.Q = persp.Q
        self.persp = persp

        self.H = {}
        
        if verbose > 0:
            print(f'  Number of views : {self.K}')
            print(f'  Number of points : {self.N}')
            print(f'  Embedding dimension : {self.persp.dimX}')
            print(f'  Projection dimension : {self.persp.dimY}')
            #print(f'  Root-mean-squared of D : {self.D_rms:0.2e}\n')

    def setup_visualization(self,visualization='mds',**kwargs):
        assert visualization in ['mds','tsne']
        self.visualization_method = visualization

        if visualization is 'mds':
            visualization_class = mds2.MDS
        elif visualization is 'tsne':
            visualization_class = tsne.TSNE

        self.visualization= []
        for k in range(self.K):
            self.visualization.append(visualization_class(self.D[k],
                                                          self.persp.dimY,
                                                          **kwargs))
        
        def cost_function(X,Q,Y=None):
            if Y is None:
                Y = self.persp.compute_Y(X,Q=Q)
            cost = 0
            for k in range(self.K):
                cost += self.visualization[k].cost_function(Y[k])**2
            return math.sqrt(cost)
        self.cost_function = cost_function

        def cost_function_k(X,q,k,y=None):
            if y is None:
                y = self.persp.compute_Y(X,q=q)
            cost_k = self.visualization[k].cost_function(y)
            return cost_k
        self.cost_function_k = cost_function_k

        def cost_function_all(X,Q,Y=None):
            if Y is None:
                Y = self.persp.compute_Y(X,Q=Q)
            cost = 0; individual_cost = np.zeros(self.K)
            for k in range(self.K):
                cost_k = self.visualization[k].f(Y[k])
                cost += cost_k
                individual_cost[k] = cost_k
            return cost, individual_cost
        self.cost_function_all = cost_function_all
        
        self.cost_function = cost_function
        
        if self.persp.family == 'linear':

            def F(X,Q,batch_number=None,batch_size=None,Y=None):
                if Y is None:
                    Y = self.persp.compute_Y(X,Q=Q)
                if batch_size is None and batch_number is None:
                    batches = None
                else:
                    if isinstance(batch_number,int):
                        batch_size = math.ceil(self.N/batch_number)
                    elif isinstance(batch_size,int):
                        batch_number = math.ceil(self.N/batch_size)
                    else:
                        sys.exit('wrong batch_size and batch_number')
                    indices = list(range(self.N)); random.shuffle(indices)
                    batches = [list(indices[j*batch_size:(j+1)*batch_size]) for\
                               j in range(batch_number)]
                    
                cost = 0
                dX = np.zeros((self.N,self.persp.dimX))
                dQ = []
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k])#,batches)
                    cost += costk**2
                    dX += dYk @ Q[k]
                    dQ.append(dYk.T @ X)
                cost = math.sqrt(cost)
                return (cost,[dX]+dQ)
            self.F = F

            def FX(X,Q,Y=None,**kwargs):
                if Y is None:
                    Y = self.persp.compute_Y(X,Q=Q)
                cost = 0
                dX = np.zeros((self.N,self.persp.dimX))
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],**kwargs)
                    cost += costk
                    dX += dYk @ Q[k]
                return (cost,dX)
            self.FX = FX

            def FQ(X,Q,batch_number=None,batch_size=None,Y=None):
                if Y is None:
                    Y = self.persp.compute_Y(X,Q=Q)
                    
                if batch_size is None and batch_number is None:
                    batches = None
                else:
                    if isinstance(batch_number,int):
                        batch_size = math.ceil(self.N/batch_number)
                    elif isinstance(batch_size,int):
                        batch_number = math.ceil(self.N/batch_size)
                    else:
                        sys.exit('wrong batch_size and batch_number')
                    indices = list(range(self.N)); random.shuffle(indices)
                    batches = [list(indices[j*batch_size:(j+1)*batch_size]) for\
                               j in range(batch_number)]
                    
                cost = 0
                dQ = []
                for k in range(self.K):
                    costk, dYk = self.visualization[k].F(Y[k],batches)
                    cost += costk
                    dQ.append(dYk.T @ X)
                return (cost,dQ)
            self.FQ = FQ
            
        else:

            def gradient_X(X,Q,Y=None):
                pgradient = self.proj.compute_gradient(X[0],params_list=Q)
                print(pgradient[0])
                if Y is None:
                    Y = self.proj.project(X,params_list=Q)
                gradient = np.zeros((self.N,self.dimX))
                for k in range(self.K):
                    gradient += self.visualization[k].gradient_function(Y[k]) \
                                @ pgradient[k]
                return gradient
            self.gradient_X = gradient_X

            def partial_X(X,Q,n,Y=None):
                pgradient = self.proj.compute_gradient(X[n],params_list=Q)
                if Y is None:
                    Y = self.proj.project(X,params_list=Q)
                partial = np.zeros(self.dimX)
                for k in range(self.K):
                    partial += self.visualization[k].partial_function(Y[k],n) \
                               @ pgradient[k]
                return partial
            self.partial_X = partial_X

            def batch_X(X,Q,nn,Y=None):
                pgradient = self.proj.compute_gradient(X[0],params_list=Q)
                if Y is None:
                    Y_batch = self.proj.project(X[nn],params_list=Q)
                else:
                    Y_batch = Y[nn]
                gradient_batch = np.zeros((len(nn),self.dimX))
                for k in range(self.K):
                    gradient_batch += self.visualization[k].batch_function(Y)
                return gradient_batch ######
            self.batch_X = batch_X

    def initialize_Q(self, **kwargs):
        """\
        Set initial parameters for perspective functions.
        """
        if self.verbose > 0:
            print('- Multiview.initialize_Q():')
        self.Q = self.persp.generate_Q(number=self.K,**kwargs)
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
            print('- Multiview.initialize_X():')

        if X0 is not None:
            if self.verbose > 0:
                print('  method : X0 given')
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.N,self.persp.dimX)
            self.X = X0
        else:
            if self.verbose > 0:
                print('  method : ',method)
            if method == 'random':
                self.X = misc.initial_embedding(self.N,dim=self.persp.dimX,
                                                radius=1)
                                                #radius=self.D_rms,**kwargs)
            elif method == 'mds':
                D = np.average(self.D,axis=0)
                vis = mds.MDS(D,dim=self.persp.dimX)
                vis.initialize()
                vis.optimize(max_iters=max_iters,**kwargs)
                self.X = vis.X
        self.update()
        self.X0 = self.X.copy()

    def update(self,H=None):
        if hasattr(self,'X') and hasattr(self,'Q'):
            self.Y = self.persp.compute_Y(self.X,Q=self.Q)

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

    def optimize_X(self, agd=True, approx=0.5, lr=5.,
                   **kwargs):
        if self.verbose > 0:
            print('- Multiview.optimize_X():')

        if approx is not None:
            if self.verbose > 0:
                print('  method : stochastic gradient descent')
                print('  approx =',approx)
            F = lambda X: self.FX(X,self.Q,approx=approx)
            self.X, H = gd.mgd(self.X,F,lr=lr,**kwargs)
            self.update(H=H)
        if agd is True:
            F = lambda X: self.FX(X,self.Q)
            if self.verbose > 0:
                print('  method : exact gradient & adaptive gradient descent')
            self.X, H = gd.agd(self.X,F,**kwargs,**self.H)
            self.update(H=H)

        if self.verbose > 0:
            print(f'  Final stress : {self.cost:0.2e}')

    def optimize_Q(self,batch_size=None,batch_number=None,lr=0.01,**kwargs):
        if self.verbose > 0:
            print('- Multiview.optimize_Q():')

        F = lambda Q: self.FQ(self.X,Q,batch_number=batch_number,
                              batch_size=batch_size)
        if batch_number is None and batch_size is None:
            self.Q, H = gd.cagd(self.Q,F,**kwargs)
        else:
            self.Q, H = gd.mgd(self.Q,F,lr=lr,**kwargs)
        self.update(H=H)

        if self.verbose > 0:
            print(f'  Final stress : {self.cost:0.2e}')

    def optimize_all(self,agd=True,batch_size=None,batch_number=None,lr=0.01,
                     **kwargs):
        if self.verbose:
            print('- Multiview.optimize_all(): ')

        p = [None]+[self.persp.c]*self.K
        if batch_number is not None or batch_size is not None:
            XQ = [self.X]+self.Q;
            F = lambda XQ: self.F(XQ[0],XQ[1::],batch_number=batch_number,
                                  batch_size=batch_size)
            XQ, H = gd.mgd(XQ,F,lr=lr,**kwargs)
            self.X = XQ[0]; self.Q = XQ[1::]; self.update(H=H)
        if agd is True:
            XQ = [self.X]+self.Q;
            F = lambda XQ: self.F(XQ[0],XQ[1::])
            XQ, H = gd.cagd(XQ,F,**kwargs,**self.H)
            self.X = XQ[0]; self.Q = XQ[1::]; self.update(H=H)

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
                
    def figureX2(self,title='Final embedding',perspectives=True,
                labels=None,edges=None,colors=None,plot=True):
        if labels is None:
            labels = self.labels
        if self.persp.dimX == 2:
            fig = plt.figure(1)
            plt.plot(self.X[:,0],self.X[:,1],'o',c=colors)
            plt.title(title+f', normalized cost : {self.ncost:0.2e}')
            if plot is True:
                plt.draw()
                plt.pause(0.1)
            return fig
        else:
            plt.figure()
            axes = plt.axes(projection='3d')
            if perspectives is True:
                max0=np.max(np.abs(self.X[:,0]))
                max1=np.max(np.abs(self.X[:,1]))
                max2=np.max(np.abs(self.X[:,2]))
                maxes = np.array([max0,max1,max2])
                for k in range(self.K):
                    Q = self.Q[k]
                    q = np.cross(Q[0],Q[1])
                    ind = np.argmax(q/maxes)
                    m = maxes[ind]
                    axes.plot([0,m*q[0]],[0,m*q[1]],[0,m*q[2]],'-',linewidth=3,
                              color='black')
            if edges is not None:
                if isinstance(edges,numbers.Number):
                    edges = edges-self.D
                for k in range(self.K):
                    for i in range(self.N):
                        for j in range(i+1,self.N):
                            if edges[i,j] > 0:
                                axes.plot([self.X[i,0],self.X[j,0]],
                                             [self.X[i,1],self.X[j,1]],
                                             [self.X[i,2],self.X[j,2]],'-',
                                          linewidth=0.25,color='blue')#,l='b')
            axes.scatter3D(self.X[:,0],self.X[:,1],self.X[:,2],c=colors)
            #axs.set_aspect(1.0)
            plt.title(title+f', normalized cost = {self.ncost:0.2e}')
            if plot is True:
                plt.draw()
                plt.pause(0.1)

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
        plots.plot_cost(self.H['cost'],self.H['steps'],title=title,ax=ax)
        if plot is True:
            plt.draw()
            plt.pause(0.2)

    def figureHY(self,title='multiview computation & embedding',colors=True,
               edges=False,plot=True):
        assert self.persp.dimY >= 2
        fig,axs = plt.subplots(1,self.K+1)
        plt.suptitle(title+f', cost : {self.cost:0.2e}')
        self.figureH(ax=axs[0])
        self.figureY(ax=axs[1::],colors=colors,edges=edges)
        if plot is True:
            plt.draw()
            plt.pause(1)
    
##### TESTS #####

def example_disk(N=100):
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    persp = perspective.Persp()
    persp.fix_Q(number=3,special='standard')
    D = multigraph.from_perspectives(X,persp)
    mv = MPSE(D,persp=persp,verbose=1)
    mv.setup_visualization(visualization='mds')
    mv.initialize_X(verbose=1)
    mv.optimize_X(batch_size=10,max_iters=50,verbose=1)
    mv.figureX(save='hola')
    mv.figureY()
    mv.figureH()
    mv.figureHY()
    plt.show()

def example_disk_Q(N=100):
    X = misc.disk(N,dim=3)
    persp = perspective.Persp()
    persp.fix_Q(number=3,special='standard')
    D = multigraph.from_perspectives(X,persp)
    mv = MPSE(D,persp=persp,verbose=1)
    mv.setup_visualization(visualization='mds')
    mv.initialize_Q(random='orthogonal')
    mv.initialize_X(X0=X)
    mv.optimize_Q(verbose=2)
    mv.figureHY()
    plt.show()

def example_disk_all(N=100):
    X = misc.disk(N,dim=3); labels=misc.labels(X)
    persp = perspective.Persp()
    persp.fix_Q(number=3,special='standard')
    D = multigraph.from_perspectives(X,persp)
    mv = MPSE(D,persp=persp,verbose=1)
    mv.setup_visualization(visualization='mds')
    mv.initialize_Q()
    mv.initialize_X()
    mv.optimize_all(agd=True)
    mv.figureX(plot=True)
    mv.figureY(plot=True)
    mv.figureH()
    plt.show()

def example_binomial(N=100,K=2):
    persp = perspective.Persp()
    for p in [0.05,0.1,0.5,1.0]:
        D = multigraph.binomial(N,p,K=K)
        mv = MPSE(D,persp=persp,verbose=1)
        mv.setup_visualization(visualization='mds')
        mv.initialize_Q()
        mv.initialize_X()
        mv.optimize_all(agd=True,max_iters=400,min_step=1e-8)
        mv.figureX(plot=True)
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
    proj = perspective.Proj(dimX=2,dimY=2)
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
    #example_disk(30)
    #example_disk_Q(30)
    #example_disk_all(N=30)
    #example_binomial(N=30,K=3)
    #noisy()
    #noisy_combine()
    #test_mds0()
    #test_mds123()#save_data=True)
    example_random_graph_perspectives(N=100)
    #xyz()
### Older Tests ###

def test_mds123(save_data=False):
    Y1 = np.load('examples/123/true1.npy'); D1 = distances.coord2dist(Y1)
    Y2 = np.load('examples/123/true2.npy'); D2 = distances.coord2dist(Y2)
    Y3 = np.load('examples/123/true3.npy'); D3 = distances.coord2dist(Y3)
    D = [D1,D2,D3]
    
    Q = np.load('examples/123/params.npy')
    proj = perspective.Proj()
    proj.set_params_list(params_list=Q)
    
    mv = Multiview(D,persp=proj)
    mv.setup_visualization('mds')
    mv.initialize_X(verbose=1)
    mv.optimize_X(algorithm='cdm',learning_rate=0.005,iterations=300)
    mv.figureX(); mv.figureY(); plt.show()

    if save_data:
        np.save('examples/123/computed123.npy',mv.X)
        np.save('examples/123/computed1.npy',mv.Y[0])
        np.save('examples/123/computed2.npy',mv.Y[1])
        np.save('examples/123/computed3.npy',mv.Y[2])
    
