import os, sys
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, directory)

import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance
from scipy.spatial.distance import squareform

import misc, setup, multigraph, gd, projections, mds, tsne, plots

class MPSE(object):
    """\
    Class to set up and produce multi-perspective simultaneous embeddings.
    """

    def __init__(self, data, weights=None, data_args=None,
                 fixed_embedding=None, fixed_projections=None,
                 initial_embedding=None, initial_projections=None,
                 visualization_method='mds', visualization_args={},
                 total_cost_function='rms',
                 embedding_dimension=3, image_dimension=2,
                 projection_family='linear',projection_constraint='orthogonal',
                 hidden_samples=None,
                 sample_labels=None, perspective_labels=None,
                 sample_colors=None, image_colors=None,
                 verbose=0, indent='',
                 **kwargs):
        """\
        Initializes MPSE object.

        Parameters
        ----------

        data : list, length (n_perspectives)
        List containing distance/dissimilarity/feature data for each 
        perspective. Each array can be of the following forms:
        1) A 1D condensed distance array
        2) A square distance matrix
        3) An array containing features
        ***4) A dictionary describing a graph

        weights : None or string or array or list
        If visualization allows for it, weights to be used in computation of
        cost/gradiant of each perspective.
        IF a list is given, then the list must have length equal to the number
        of perspectives. Otherwise, it is assumed that the given weights are the
        same for all perspectives.
        The possible weights are described in setup.setup_weights. These are:
        1) None : no weights are used
        2) string : method to compute weights based on distances
        3) function : function to compute weights based on distances
        4) array : array containing pairwise weights or node weights, depending
        on size (must be of length of distances or of samples).

        data_args : dictionary (optional) or list
        Optional arguments to pass to distances.setup().
        If a list is passed, then the length must be the number of perspectives
        and each element must be a dictionary. Then, each set of distances will
        be set up using a different set of arguments.

        fixed_embedding : array
        If an array is given, this is assumed to be the true embedding and
        by default optimization is done w.r.t. the projections only.

        fixed_projections : list
        If a list is given, this is assumed to be the true projections and by
        default optimization is done w.r.t. the embedding coordinates only.

        initial_embedding : array
        If given, this is the initial embedding used.

        initial_projections : list
        If given, this is the initial projections used.

        visualization_method : str
        Visualization method. Current options are 'mds' and 'tsne'.
        The visualization method can be different for different perspectives, 
        by passing a list of visualization methods instead.

        visualization_args : dict
        Dictionary with arguments to pass to each visualization method.
        Different arguments can be passed to different visualization methods
        by passing a list of dictionaries instead.

        embedding_dimension : int
        Dimension of embedding.

        image_dimension : int
        Dimension of image (after projection). Each perspective can have a
        different image dimension, by specifying a list instead.

        projection_family : str
        Projection family. Options are 'linear'.

        projection_constraint : str
        Constraints on projection family. Options are None, 'orthogonal',
        'similar'.

        embedding_dimension : int > 0
        Dimension of the embedding.
        Alternative name: embedding_dimension

        projection_dimension : int or array
        Dimension of projections. Can be different for each perspective.
 
        persp : Object instance of projections.Persp class or int > 0.
        Describes set of allowed projection functions and stores list of
        projection parameters. See perspective.py. If instead of a Persp object
        a positive integer int is given, then it is assumed that 
        embedding_dimension=image_dimension=int
        and that all projections are the identity.

        sample_labels : list (optional)
        List containing labels of samples (used in plots).

        sample_colors : array (optional)
        Array containing color value of samples (used in plots).

        image_colors : array-like, shape (n_perspectives, n_samples)
        Colors for each image.
        """
        self.verbose, self.indent = verbose, indent
        if verbose > 0:
            print(indent+'mview.MPSE():')

        ##set up sets of distances from data
        self.distances = setup.setup_distances_from_multiple_perspectives(
            data, data_args)
        self.n_perspectives = len(self.distances)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances[0])

        ##set up weights from data
        if isinstance(weights,list) or isinstance(weights, np.ndarray):
            assert len(weights) == self.n_perspectives
            self.weights = weights
        else:
            self.weights = [weights]*self.n_perspectives
        for i in range(self.n_perspectives):
            self.weights[i] = setup.setup_weights(self.distances[i], \
                                self.weights[i], min_weight = 0)

        ##set up parameters
        self.embedding_dimension = embedding_dimension
        self.image_dimension = image_dimension
        self.projection_family = projection_family
        self.projection_constraint = projection_constraint
        proj = projections.PROJ(embedding_dimension,image_dimension,
                                projection_family,projection_constraint)
        self.proj = proj

        ##set up hidden samples
        if hidden_samples is not None:
            assert isinstance(hidden_samples, list)
            assert len(hidden_samples) == self.n_perspectives
        self.hidden_samples = hidden_samples

        if verbose > 0:
            print(indent+'  data details:')
            print(indent+f'    number of perspectives : {self.n_perspectives}')
            print(indent+f'    number of samples : {self.n_samples}')
            print(indent+'  visualization details:')
            print(indent+'    embedding dimension :',self.embedding_dimension)
            print(indent+f'    image dimension : {self.image_dimension}')
            print(indent+f'    visualization type : {visualization_method}')

        #setup sample labels:
        if sample_labels is not None:
            assert len(sample_labels) == self.n_samples
        self.sample_labels = sample_labels
        #setup perspective labels:
        if perspective_labels is None:
            perspective_labels = range(1,self.n_perspectives+1)
        else:
            assert len(perspective_labels) == self.n_perspectives
        self.perspective_labels = perspective_labels
        
        #setup colors:
        self.sample_colors = sample_colors
        self.image_colors = image_colors

        #setup visualization instances:
        self.visualization_instances = []
        self.visualization_method = visualization_method
        if isinstance(visualization_method,str):
            visualization_method = [visualization_method]*self.n_perspectives
        if isinstance(visualization_args,dict):
            visualization_args = [visualization_args]*self.n_perspectives
        for i in range(self.n_perspectives):
            assert visualization_method[i] in ['mds','tsne']
            if self.verbose > 0:
                print('  setup visualization instance for perspective',
                      self.perspective_labels[i],':')
            if visualization_method[i] is 'mds':
                vis = mds.MDS(self.distances[i],
                              weights = self.weights[i],
                              embedding_dimension=self.image_dimension,
                              verbose=self.verbose, indent=self.indent+'    ',
                              **visualization_args[i])
            elif visualization_method[i] is 'tsne':
                vis = tsne.TSNE(self.distances[i],
                                embedding_dimension=self.image_dimension,
                                verbose=self.verbose, indent=self.indent+'    ',
                                **visualization_args[i])
            self.visualization_instances.append(vis)
        self.visualization = self.visualization_instances

        #setup objectives:
        if total_cost_function == 'rms':
            self.total_cost_function = lambda individual_costs : \
                np.sqrt(np.sum(individual_costs**2)/self.n_perspectives)
        else:
            assert callable(total_cost_function)
            self.total_cost_function = total_cost_function
        def cost_function(X,Q,Y=None,**kwargs):
            if Y is None:
                Y = self.proj.project(Q,X)
            individual_costs = np.zeros(self.n_perspectives)
            for k in range(self.n_perspectives):
                individual_costs[k] = \
                    self.visualization[k].objective(Y[k],**kwargs)
            cost = self.total_cost_function(individual_costs)
            return cost, individual_costs
        self.cost_function = cost_function

        #setup gradient function:
        if self.projection_family == 'linear':
            def gradient(embedding,projections,batch_size=None,indices=None,
                         return_embedding=True,return_projections=True,
                         return_cost=True, return_individual_costs=False):
                """\
                Returns MPSE gradient(s), along with cost and individual costs 
                (optional).

                Parameters
                ----------

                embedding : numpy array
                Current embedding.

                projections : numpy array
                Current projections (as a single array).

                return_embedding : boolean
                If True, returns MPSE gradient w.r.t. embedding.

                return_projections : boolean
                If True, returns MPSE gradient w.r.t. projections. 

                return_cost : boolean
                If True, returns MPSE cost.

                return_individual_costs : boolean
                If True, returns individual embedding costs.
                """
                if return_embedding:
                    dX = np.zeros(embedding.shape)
                if return_projections:
                    dQ = []
                individual_costs = np.empty(self.n_perspectives)
                Y = self.proj.project(projections,embedding)
                for k in range(self.n_perspectives):
                    dY_k, cost_k = self.visualization[k].gradient(
                        Y[k],batch_size=batch_size,indices=indices)
                    individual_costs[k] = cost_k
                    if return_embedding:
                        dX += dY_k @ projections[k]
                    if return_projections:
                        dQ.append(dY_k.T @ embedding)
                if return_embedding:
                    dX /= self.n_perspectives
                cost = self.total_cost_function(individual_costs)
                if return_embedding is False:
                    grad = np.array(dQ)
                elif return_projections is False:
                    grad = dX
                else:
                    grad = [dX,np.array(dQ)]
                if return_individual_costs:
                    return grad, cost, individual_costs
                else:
                    return grad, cost
            self.gradient = gradient
        else:
            def gradient_X(X,Q,Y=None):
                pgradient = self.proj.compute_gradient(X[0],params_list=Q)
                if Y is None:
                    Y = self.proj.project(X,params_list=Q)
                gradient = np.zeros((self.n_samples,self.embedding_dimension))
                for k in range(self.n_perspectives):
                    gradient += self.visualization[k].gradient(Y[k]) \
                                @ pgradient[k]
                return gradient
            self.gradient_X = gradient_X

        #set up initial embedding and projections (fixed optional):
        if verbose > 0:
            print(indent+'  initialize:')
        #set fixed and initial embedding:
        if fixed_embedding is not None:
            if verbose > 0:
                print(indent+'    fixed embedding : True')
            self.embedding = fixed_embedding
            self.initial_embedding = fixed_embedding
            self.fixed_embedding = True
        else:
            if verbose > 0:
                print(indent+'    fixed embedding : False')
            if initial_embedding is None:
                if verbose > 0:
                    print(indent+'    initial embedding : random')
                self.initial_embedding = misc.initial_embedding(
                    self.n_samples,dim=self.embedding_dimension, radius=1)
            else:
                assert isinstance(initial_embedding,np.ndarray)
                assert initial_embedding.shape == (
                    self.n_samples, self.embedding_dimension)
                if verbose > 0:
                    print(indent+'    initial embedding : given')
                self.initial_embedding = initial_embedding
            self.embedding = self.initial_embedding
            self.fixed_embedding = False
        #set fixed and initial projections:
        if fixed_projections is not None:
            if isinstance(fixed_projections,str):
                fixed_projections = self.proj.generate(number= \
                            self.n_perspectives,method=fixed_projections)
            self.projections = fixed_projections
            self.initial_projections = fixed_projections
            self.fixed_projections = True
            if verbose > 0:
                print(indent+'    fixed projections : True')
        else:
            if verbose > 0:
                print(indent+'    fixed projections : False')
            if initial_projections is None:
                if verbose > 0:
                    print(indent+'    initial projections : random')
                self.initial_projections = self.proj.generate(
                    number=self.n_perspectives, **kwargs)
            else:
                if verbose > 0:
                    print(indent+'    initial projections : given')
                if isinstance(initial_projections,str):
                    initial_projections = self.proj.generate(number= \
                            self.n_perspectives,method=initial_projections)
                self.initial_projections = initial_projections
            self.projections = self.initial_projections
            self.fixed_projections = False        

        self.initial_cost = None
        self.initial_individual_cost = None
        self.computation_history = []
        self.time = 0
        self.update(**kwargs)

    def update(self,**kwargs):
        self.images = self.proj.project(self.projections,self.embedding)
        self.cost, self.individual_cost = self.cost_function(
            self.embedding,self.projections,Y=self.images,**kwargs)
        if self.initial_cost is None:
            self.initial_cost = self.cost
            self.initial_individual_cost = self.individual_cost
        else:
            self.time = self.computation_history[-1]['time']

    def smart_initialize(self,max_iter=[50,30],lr=[1,0.1],
                         batch_size=10,**kwargs):
        """\
        Computes an mds embedding (dimension embedding_dimension) of the 
        combined distances. Only works when self.Q_is_fixed is False (as this 
        is unnecessary otherwhise).

        Parameters :

        X0 : None or array
        Optional initial embedding (used to compute mds embedding)
        
        Q0 : None or list of arrays
        Optional initial projection parameters.
        """
        assert self.fixed_embedding is False
        assert self.fixed_projections is False
    
        if self.verbose > 0:
            print(self.indent+'  MPSE.smart_initialize():')

        distances = np.sum(self.distances,axis=0)/self.n_perspectives
        if self.weights is not None and self.weights[0] is not None:
            weights = np.product(self.weights,axis=0)
        else:
            weights = None
        vis = mds.MDS(distances,dim=self.embedding_dimension,min_grad=1e-4,
                      indent=self.indent+'    ',
                      initial_embedding=self.embedding,
                      weights = weights,
                      verbose=self.verbose)
        vis.gd(batch_size=batch_size, max_iter=max_iter[0],lr=lr[0],**kwargs)
        self.embedding = vis.X
        H = vis.computation_history[0]
        H['fixed_embedding'] = False
        H['fixed_projections'] = True
        self.computation_history.append(H)

        def Xi():
            indices = np.arange(self.n_samples)
            np.random.shuffle(indices)
            xi = {
                'indices' : indices
            }
            return xi
        F = lambda Q, indices : self.gradient(self.embedding,Q,
                                              batch_size=batch_size,
                                              return_embedding=False)
        Q0 = np.array(self.projections)
        self.projections, H = gd.single(Q0,F,Xi=Xi,p=self.proj.restrict,
                              max_iter=max_iter[1],lr=lr[1],
                              verbose=self.verbose,indent=self.indent+'    ',
                              **kwargs)
        H['fixed_embedding'] = True
        H['fixed_projections'] = False
        self.computation_history.append(H)
        return
                
    def gd(self, batch_size=None, lr=None, fixed_projections='default',
           fixed_embedding='default', **kwargs):

        if fixed_projections == 'default':
            fixed_projections = self.fixed_projections
        if fixed_embedding == 'default':
            fixed_embedding = self.fixed_embedding
            
        assert batch_size is None or isinstance(batch_size,int)
        assert fixed_embedding is False or fixed_projections is False

        if lr is None:
            if self.visualization_method == 'mds':
                if fixed_projections:
                    lr = 1
                elif fixed_embedding:
                    lr = 0.01
                else:
                    lr = [1,0.01]
            elif self.visualization_method == 'tsne':
                if fixed_projections:
                    lr = 100
                elif fixed_embedding:
                    lr = 10
                else:
                    lr = [100,10]
            else:
                lr = [1,1]
                
        if self.verbose > 0:
            print(self.indent+'  MPSE.gd():')
            print(self.indent+f'      initial stress : {self.cost:0.2e}')

        if fixed_projections:
            if self.verbose > 0:
                print(self.indent+'      mpse method : fixed projections')
            if batch_size is None or batch_size >= self.n_samples:
                Xi = None
                F = lambda X : self.gradient(X,self.projections,
                                             return_projections=False)
            else:
                if self.verbose > 0:
                    print(self.indent+'      batch size :',batch_size)
                def Xi():
                    indices = np.arange(self.n_samples)
                    np.random.shuffle(indices)
                    xi = {
                        'indices' : indices
                    }
                    return xi
                F = lambda X, indices: self.gradient(X,self.projections,
                                                     batch_size=batch_size,
                                                     indices=indices,
                                                     return_projections=False)
            self.embedding, H = gd.single(
                self.embedding,F,Xi=Xi, verbose=self.verbose, lr=lr,
                indent=self.indent+'    ',**kwargs)
            H['fixed_projections'] = True
            H['fixed_embedding'] = False
            self.computation_history.append(H)
        elif fixed_embedding:
            if self.verbose > 0:
                print(self.indent+'      mpse method : fixed embedding')
            if batch_size is None or batch_size >= self.n_samples:
                Xi = None
                F = lambda Q : self.gradient(self.embedding,Q,
                                             return_embedding=False)
            else:
                if self.verbose > 0:
                    print(self.indent+'      batch size :',batch_size)
                def Xi():
                    indices = np.arange(self.n_samples)
                    np.random.shuffle(indices)
                    xi = {
                        'indices' : indices
                    }
                    return xi
                F = lambda Q, indices: self.gradient(self.embedding,Q,
                                                     batch_size=batch_size,
                                                     indices=indices,
                                                     return_embedding=False)
            Q0 = np.array(self.projections)
            self.projections, H = gd.single(Q0,F,Xi=Xi,p=self.proj.restrict,
                                  verbose=self.verbose,lr=lr,
                                  indent=self.indent+'    ',**kwargs)
            H['fixed_projections'] = False
            H['fixed_embedding'] = True
            self.computation_history.append(H)
        else:
            if self.verbose > 0:
                print(self.indent+'      fixed : None')
            if batch_size is None or batch_size >= self.n_samples:
                Xi = None
                F = lambda params : self.gradient(params[0],params[1])
            else:
                if self.verbose > 0:
                    print(self.indent+'      batch size :',batch_size)
                def Xi():
                    indices = np.arange(self.n_samples)
                    np.random.shuffle(indices)
                    xi = {
                        'indices' : indices
                    }
                    return xi
                F = lambda params, indices: self.gradient(params[0],params[1],
                                                     batch_size=batch_size,
                                                     indices=indices)
            params = [self.embedding,np.array(self.projections)]
            params, H = gd.multiple(params,F,Xi=Xi,p=[None,self.proj.restrict],
                                    verbose=self.verbose,lr=lr,
                                    indent=self.indent+'    ',**kwargs)
            self.embedding = params[0]
            self.projections = params[1]
            H['fixed_projections'] = False
            H['fixed_embedding'] = False
            self.computation_history.append(H)
            
        self.update()

        if self.verbose > 0:
            print(self.indent+f'    final cost : {self.cost:0.2f}')
            costs = ', '.join(f'{x:0.2f}' for x in self.individual_cost)
            print(self.indent+f'    individual costs : {costs}')

    def optimized(self, iters=[40,40,40,100], **kwargs):
        "find optimal solution"
        if self.verbose > 0:
            print(self.indent+'  MPSE.optimized():')
        self.gd(batch_size=self.n_samples//20, max_iter=iters[0], scheme='mm')
        self.gd(batch_size=self.n_samples//10, max_iter=iters[1], scheme='mm')
        self.gd(batch_size=self.n_samples//5, max_iter=iters[2], scheme='mm')
        self.gd(max_iter=iters[3],scheme='bb')
 
    def plot_embedding(self,title=None,perspectives=True,edges=None,colors=True,
                plot=True,ax=None,**kwargs):

        if perspectives is True:
            perspectives = []
            for k in range(self.n_perspectives):
                Q = self.projections[k]
                q = np.cross(Q[0],Q[1])
                perspectives.append(q)
        else:
            perspectives = None
            
        if edges is not None:
            if isinstance(edges,numbers.Number):
                edges = edges-self.D
                
        if colors is True:
            colors = self.sample_colors
        if isinstance(colors, int):
            assert colors in range(self.n_samples)
            colors = squareform(self.distances[0])[colors]
            
        plots.plot3D(self.embedding,perspectives=perspectives,edges=edges,
                     colors=colors,title=title,ax=ax,**kwargs)

    def plot_images(self,title=None,edges=None,
                colors=True,plot=True,
                ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,self.n_perspectives,
                                   figsize=(3*self.n_perspectives,3))
        else:
            plot = False

        if edges is None:
            edges = [None]*self.n_perspectives
        else:
            edges = edges

        #setup colors
        if colors is True:
            colors = self.image_colors
        if colors is None:
            colors = self.sample_colors
            
        for k in range(self.n_perspectives):

            if isinstance(colors,list) and len(colors)==self.n_perspectives:
                colors_k = colors[k]
            else:
                colors_k = colors
            if isinstance(colors_k, int):
                assert colors_k in range(self.n_samples)
                colors_k = scipy.spatial.distance.squareform(self.distances[k])[colors_k]

            plots.plot2D(self.images[k],edges=edges[k],colors=colors_k,ax=ax[k],
                    weight=self.weights[k], **kwargs)
            #ax[k].set_xlabel('individual cost:'+ f'{self.individual_cost[k]}')
        plt.suptitle(title)
        if plot is True:
            plt.draw()
            plt.pause(1.0)
    
    def plot_computations(self,title='computations',plot=True,ax=None):
        if self.fixed_embedding is True or self.fixed_projections is True:
            if ax is None:
                fig, ax = plt.subplots(1,2,figsize=(6,3))
                fig.subplots_adjust(top=0.8)
            costs = np.array([])
            grads = np.array([])
            lrs = np.array([])
            steps = np.array([])
            iterations=0; markers = []
            for H in self.computation_history:
                if iterations != 0:
                    ax.axvline(x=iterations-1,ls='--',c='black',lw=.5)
                iterations += H['iterations']
                costs = np.concatenate((costs,H['costs']))
                grads = np.concatenate((grads,H['grads']))
                lrs = np.concatenate((lrs,H['lrs']))
                steps = np.concatenate((steps,H['steps']))
            ax[0].semilogy(costs,label='stress',linewidth=3)
            ax[1].semilogy(grads,label='gradient size')
            ax[1].semilogy(lrs,label='learning rate')
            ax[1].semilogy(steps,label='step size')
            ax[1].legend()
            ax[0].set_xlabel('iterations')
            ax[0].set_ylabel('stress')
            ax[1].set_xlabel('iterations')
            ax[1].set_ylabel('size')
            ax[0].set_title('MPSE stress')
            ax[1].set_title('embedding parameters')
            if plot is True:
                plt.draw()
                plt.pause(1.0)
        else:
            if ax is None:
                fig, ax = plt.subplots(1,3,figsize=(3*3,3))
                fig.subplots_adjust(top=0.80)
            costs = np.array([])
            grads_Q = np.array([])
            lrs_Q = np.array([])
            steps_Q = np.array([])
            grads_X = np.array([])
            lrs_X = np.array([])
            steps_X = np.array([])
            iterations=0; markers = []
            
            for H in self.computation_history:
                if iterations != 0:
                    ax[0].axvline(x=iterations-1,ls='--',c='black',lw=.5)
                    ax[1].axvline(x=iterations-1,ls='--',c='black',lw=.5)
                    ax[2].axvline(x=iterations-1,ls='--',c='black',lw=.5)
                iterations += H['iterations']
                costs = np.concatenate((costs,H['costs']))
                if H['fixed_projections']:
                    grads_X = np.concatenate((grads_X,H['grads']))
                    lrs_X = np.concatenate((lrs_X,H['lrs']))
                    steps_X = np.concatenate((steps_X,H['steps']))
                    grads_Q = np.concatenate((grads_Q,[None]*H['iterations']))
                    lrs_Q = np.concatenate((lrs_Q,[None]*H['iterations']))
                    steps_Q = np.concatenate((steps_Q,[None]*H['iterations']))
                elif H['fixed_embedding']:
                    grads_X = np.concatenate((grads_X,[None]*H['iterations']))
                    lrs_X = np.concatenate((lrs_X,[None]*H['iterations']))
                    steps_X = np.concatenate((steps_X,[None]*H['iterations']))
                    grads_Q = np.concatenate((grads_Q,H['grads']))
                    lrs_Q = np.concatenate((lrs_Q,H['lrs']))
                    steps_Q = np.concatenate((steps_Q,H['steps']))
                else:
                    grads_X = np.concatenate((grads_X,H['grads'][:,0]))
                    lrs_X = np.concatenate((lrs_X,H['lrs'][:,0]))
                    steps_X = np.concatenate((steps_X,H['steps'][:,0]))
                    grads_Q = np.concatenate((grads_Q,H['grads'][:,1]))
                    lrs_Q = np.concatenate((lrs_Q,H['lrs'][:,1]))
                    steps_Q = np.concatenate((steps_Q,H['steps'][:,1]))
                    
            ax[0].semilogy(costs,linewidth=3)
            ax[0].set_title('MPSE stress')                                            

            ax[1].semilogy(grads_X, label='gradient size', linestyle='--')
            ax[1].semilogy(lrs_X,label='learning rate', linestyle='--')
            ax[1].semilogy(steps_X,label='step size', linestyle='--')
            ax[1].set_title('embedding stats')
            ax[1].legend()
            #ax[1].set_xlabel('iterations')
            ax[1].set_xlim([0,len(costs)])

            ax[2].semilogy(grads_Q,label='gradient size',linestyle='--')
            ax[2].semilogy(lrs_Q,label='learning rate', linestyle='--')
            ax[2].semilogy(steps_Q,label='step size',linestyle='--')
            ax[2].set_title('projections stats')
            ax[2].legend()
            ax[2].set_xlim([0,len(costs)])
                
        if plot is True:
            plt.draw()
            plt.pause(0.2)

    def save(self):
        "save results to csv files"
        location=directory+'/temp/'
        if not os.path.exists(location):
            os.makedirs(location)
            
        np.savetxt(location+'embedding.csv', self.embedding)
        for i in range(self.n_perspectives):
            np.savetxt(location+'projection_'+str(i)+'.csv',
                       self.projections[i])
            np.savetxt(location+'images_'+str(i)+'.csv',
                       self.images[i])
        if self.sample_labels is not None:
            np.savetxt(location+'sample_labels.csv', self.sample_labels,
                       fmt='%d')

def mpse_tsne(data, perplexity=30, iters=[10,10,10,100],
              verbose=2, show_plots=True, save_results = False,**kwargs):
    "Runs MPSE optimized for tsne"
    
    #load data
    if isinstance(data,str):
        import samples
        kwargs0 = kwargs
        distances, kwargs = samples.mload(data, verbose=verbose, **kwargs0)
        for key, value in kwargs0.items():
            kwargs[key] = value
        
    #start MPSE object
    mv =  MPSE(distances, visualization_method='tsne',
               visualization_args={'perplexity':perplexity}, verbose=verbose,
               indent='  ', **kwargs)
    n_samples = mv.n_samples

    #search for global minima
    if isinstance(iters,int):
        iters = [20,20,iters]
    elif len(iters)==1:
        iters = [20,20,iters[0]]
    elif len(iters)==2:
        iters = [20,iters[0],iters[1]]
    mv.gd(fixed_projections=True, max_iter=iters[0], scheme='bb')
    for i, its in enumerate(iters[1:-1]):
        batch_size = min(100//(2**i),n_samples//(2**i))
        mv.gd(batch_size=batch_size, max_iter=its, scheme='mm')     
    mv.gd(max_iter=iters[-1], scheme='bb', **kwargs)
        
    #save outputs:
    if save_results is True:
        mv.save()

    if show_plots is True:
        mv.plot_computations()
        mv.plot_embedding(title='solution')
        mv.plot_images(title='solution', **kwargs)
        plt.show()
    return mv
    
    
##### TESTS #####

def basic(dataset='disk', fixed_projections=False,
             smart_initialization=True,
             verbose=2, **kwargs):
    import samples
    data = samples.mload(dataset, **kwargs)
    if fixed_projections:
        mv = MPSE(data['D'],fixed_projections=data['Q'],verbose=verbose,
                  colors=data['colors'],
                  sample_labels = data['sample_labels'],
                  image_colors=data['image_colors'],**kwargs)
    else:
        mv = MPSE(data['D'],verbose=verbose,colors=data['colors'],
                  image_colors=data['image_colors'],
                  sample_labels=data['sample_labels'],
                  **kwargs)
    
    if smart_initialization and fixed_projections is False:
        mv.smart_initialize()
        mv.plot_embedding(title='smart initialize')
        mv.plot_images(title='smart init')

    if fixed_projections:
        mv.gd(fixed_projections=True,**kwargs)
    else:
        #mv.gd(**kwargs)
        mv.optimized(**kwargs)
    #mv.gd(**kwargs) ###

    #save outputs:
    mv.save()
        
    mv.plot_computations()
    mv.plot_embedding(title='final embeding')
    mv.plot_images()#edges=edges, labels=labels)
    plt.draw()
    plt.pause(0.2)
    plt.show()
    return mv.embedding
    
if __name__=='__main__':
    print('mview.mpse : running tests')

 #   X = basic(dataset='clusters', n_samples=400, n_clusters=2,
  #            n_perspectives=3,
   #           fixed_projections=False,
    #          visualization_method='tsne',
     #         smart_initialization=False,
      #        max_iter=200, visualization_args={'perplexity':300})

    #X = basic(dataset='florence', digits=[1,7], n_samples=500,
     #         #n_clusters=[4,4,4],
      #        #n_perspectives=3,
       #       fixed_projections=False,
        #      visualization_method='tsne',
         #     smart_initialization=False,
          #    visualization_args={'perplexity':20})

    
    mpse_tsne('equidistant')
    mpse_tsne('disk', n_perspectives=10)
    mpse_tsne('clusters', n_clusters=[3,4,2], n_perspectives=3)
    mpse_tsne('clusters2', n_clusters=2, n_perspectives=4, perplexity=80)
    mpse_tsne('florence', perplexity = 40)
    mpse_tsne('123', perplexity = 980)
    mpse_tsne('credit')
    mpse_tsne('mnist',n_samples=1000,perplexity=30)
    mpse_tsne('mnist',n_samples=1000,perplexity=100)
    mpse_tsne('phishing')
    
