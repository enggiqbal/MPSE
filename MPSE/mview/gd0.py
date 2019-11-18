import copy, random, math
import matplotlib.pyplot as plt
import numpy as np

### GD-type algorithms ###

def gd(x0,gradient,learning_rate=1.0,min_step=1e-16,max_iters=1000,**kwargs):
    """\
    Basic gradient descent algorithm

    Parameters:

    x0 : array-like
    Initial parameter array.

    gradient : function(array-like)
    Gradient function. Takes and returns array-like objects with the same shape
    as x0.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    min_step : number
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped.

    max_iters : positive integer
    Number of iterations.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation.
    """
    x = x0; i = 0; step_size = np.Inf
    while step_size > min_step and i < max_iters:
        x0 = x
        grad = gradient(x0)
        x = x0 - learning_rate*grad
        step_size = np.linalg.norm(x-x0)
        i += 1

    if step_size <= min_step:
        stop = True
    else:
        stop = False
        
    specs = {
        'iterations' : i,
        'final_step' : step_size,
        'final_gradient' : np.linalg.norm(grad),
        'stop' : stop
        }
    return x, specs

def pgd(x0,gradient,p,learning_rate=1.0,min_step=1e-16,
        max_iters=1000,**kwargs):
    """\
    Projected gradient descent algorithm.

    Parameters:

    gradient : function(array-like)
    Gradient function. Takes optimization parameters x and returns gradient of
    cost function at that point.

    p : function(array-like)
    Projection function. Takes optimization parameters x and returns projection
    of x onto the allowed subset of x.

    x0 : array-like
    Initial parameter array.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    min_step : number
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped.

    max_iters : positive integer
    Maximum number of iterations.

    Returns:

    x : array-like
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation
    """
    x = x0; i = 0; step_size = np.Inf
    while step_size > min_step and i < max_iters:
        x0 = x
        grad = gradient(x0)
        x = p(x0-learning_rate*grad)
        step_size = np.linalg.norm(x-x0)
        i += 1
        
    if step_size <= min_step:
        stop = True
    else:
        stop = False
        
    specs = {
        'iterations' : i,
        'final_step' : step_size,
        'final_gradient' : np.linalg.norm(grad),
        'stop' : stop
        }
    return x, specs

def gdm(x0,gradient,learning_rate=0.01,momentum_weight=0.9,iterations=100,
        initial_momentum=0,**kwargs):
    """\
    Basic gradient descent with momentum algorithm.

    Parameters:

    x0 : array-like
    Initial position.

    gradient : function(array-like)
    Gradient function. Takes and returns array-like objects with the same shape
    as x0.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    momentum_weight : number, 0 <= momentum_factor <= 1
    Weight assigned to previous momentum when updating momentum.

    iterations : positive integer
    Number of iterations.

    initial_momentum : array-like, shape(momentum) == shape (x0)
    Initial momentum (averaged gradient).

    Returns:

    x : array-like
    Final parameter array.

    momentum : array-like
    Final momentum array.
    """
    momentum = initial_momentum
    for i in range(iterations):
        momentum = momentum_weight*momentum+(1-momentum_weight)*gradient(x0)
        x0 -= learning_rate*momentum
    return x0, momentum

def cd(x0,partial,learning_rate=0.01,iterations=10,**kwargs):
    """\
    Coordinate descent algorithm.

    Parameters:

    x0 : array-like
    Initial position.

    partial : function(x(array-like),i(index))
    Partial gradient function. It must take a position x and an index i and 
    return the approximation of the gradient at that point using only the data
    that involves data point i.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    iterations : positive integer
    Number of iterations.

    Returns:

    x : array-like
    Final parameter array.

    Notes:

    This is similar to gradient descent, but the entries/rows x[i] of
    the parameter array x are updated one at a time, using the corresponding
    partial derivative. This is also similar to stochastic gradient descent, 
    since only one of the terms in the summation defining the loss function are 
    used, but only the appropriate entry/row is updated.
    """
    x = x0; N = len(x)
    for i in range(iterations):
        indices = random.sample(list(range(N)),N)
        for j in indices:
            x[j] -= learning_rate*partial(x0,j)
    return x

def cdm(x0,partial,learning_rate=0.01,momentum_weight=0.9,iterations=10,
        initial_momentum=0,**kwargs):
    """\
    Coordinate descent with momemtum algorithm.

    Parameters:

    x0 : array-like
    Initial position.

    partial : function(x(array-like),i(index))
    Partial gradient function. It must take a position x and an index i and 
    return the approximation of the gradient at that point using only the data
    that involves data point i.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    momentum_weight : number, 0 <= momentum_factor <= 1
    Weight assigned to previous momentum when updating momentum.

    iterations : positive integer
    Number of iterations.

    initial_momentum : array-like, shape(momentum) == shape (x0)
    Initial momentum (averaged gradient).

    Returns:

    x : array-like
    Final parameter array.

    momentum : array-like
    Final momentum array.
    """
    x = x0; N = len(x)
    if initial_momentum == 0:
        momentum = np.zeros(x0.shape)
    else:
        momentum = initial_momentum
    for i in range(iterations):
        indices = random.sample(range(N),N)
        for j in indices:
            momentum[j] = momentum_weight*momentum[j]+\
                          (1-momentum_weight)*partial(x0,j)
            x[j] -= learning_rate*momentum[j]
    return x, momentum

def bgd(x0,block_gradient,iterations=10,batch_number=None,batch_size=10,
        learning_rate=0.01,**kwargs):
    """\
    Batch (coordinate) gradient descent.

    Parameters:

    x0 : array-like
    Initial position.

    block_gradient : function(x(array-like),inds(list))
    Batch gradient function. It must take a position x and a list of indices
    inds and return the approximation of the block gradient corresponding to the
    listed indices at that point, using only the data that involves data points
    with indices in inds.
    

    iterations : int > 0
    Number of iterations.

    batch_number: int > 0
    Number of batches in which the data is divided. If batch_number is None,
    then the number of batches is determined by the parameter batch_size.

    batch_size: int > 0
    Size of batches in which the data is divided. If batch_number is specified,
    then this parameter is ignored.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    Returns:

    x : array-like
    Final parameter array.
    """
    x = x0; N = len(x)
    
    if batch_number is None:
        batch_number = math.ceil(N/batch_size)
    else:
        batch_size = math.ceil(N/batch_number)

    indices = list(range(N))
    for i in range(iterations):
        random.shuffle(indices)
        for j in range(batch_number):
            index_list = indices[j*batch_size:(j+1)*batch_size]
            x[index_list] -= learning_rate*block_gradient(x0[index_list],index_list)
    return x

def bgdm(x0,block_gradient,iterations=10,batch_number=None,batch_size=10,
         learning_rate=0.01,momentum_weight=0.9,initial_momentum=0,**kwargs):
    """\
    Batch (coordinate) gradient descent with momentum.

    Parameters:

    x0 : array-like
    Initial position.

    block_gradient : function(x(array-like),inds(list))
    Batch gradient function. It must take a position x and a list of indices
    inds and return the approximation of the block gradient corresponding to the
    listed indices at that point, using only the data that involves data points
    with indices in inds.

    iterations : int > 0
    Number of iterations.

    batch_number: int > 0
    Number of batches in which the data is divided. If batch_number is None,
    then the number of batches is determined by the parameter batch_size.

    batch_size: int > 0
    Size of batches in which the data is divided. If batch_number is specified,
    then this parameter is ignored.

    learning_rate : number
    Weight assigned to gradient term when updating position.

    momentum_weight : number, 0 <= momentum_factor <= 1
    Weight assigned to previous momentum when updating momentum.

    initial_momentum : array-like, shape(momentum) == shape (x0)
    Initial momentum (averaged gradient).

    Returns:

    x : array-like
    Final parameter array.
    """
    x = x0; N = len(x)
    if initial_momentum == 0:
        momentum = np.zeros(x0.shape)
    else:
        momentum = initial_momentum
    
    if batch_number is None:
        batch_number = math.ceil(N/batch_size)
    else:
        batch_size = math.ceil(N/batch_number)

    indices = list(range(N))
    for i in range(iterations):
        random.shuffle(indices)
        for j in range(batch_number):
            index_list = indices[j*batch_size:(j+1)*batch_size]
            momentum[index_list] = momentum_weight*momentum[index_list]+\
                          (1-momentum_weight)*block_gradient(x0[index_list],index_list)
            x[index_list] -= learning_rate*momentum[index_list]
    return x, momentum

def agd(x0,gradient,min_step=1e-15,max_iters=100,previous_x=0,previous_grad=0,
        verbose=0,**kwargs):
    """\
    Adaptive gradient descent algorithm

    Parameters:

    x0 : array-like
    Initial parameter array.

    gradient : function(array-like)
    Gradient function. Takes and returns array-like objects with the same shape
    as x0.

    iterations : positive integer
    Number of iterations.

    previous_x : array-like
    Value of Parameter array in previous iteration.

    previous_grad: array-like
    Value of gradient array in previosu iteration.

    Returns:

    x : array-like
    Final parameter array.
    """
    x = x0; i = 0; step_size = np.Inf
    while step_size > min_step and i < max_iters:
        x0 = x
        grad = gradient(x0)
        dgrad = grad-previous_grad
        norm = np.linalg.norm(dgrad)
        if norm == 0.0:
            break
        learning_rate = abs(np.sum((x0-previous_x)*dgrad)/norm**2)
        previous_x = x0
        previous_grad = grad
        x = x0 - learning_rate*grad
        step_size = np.linalg.norm(x-x0)
        i += 1
        
    specs = {
        'iterations' : i,
        'final_step' : step_size,
        'final_gradient' : np.linalg.norm(grad),
        'previous_x' : previous_x,
        'previous_grad' : previous_grad,
        }
    if step_size <= min_step or norm == 0.0:
        specs['stop'] = True
    else:
        specs['stop'] = False
    return x, specs

def pagd(x0,gradient,p,min_step=1e-15,max_iters=100,previous_x=0,
         previous_grad=0,verbose=0,**kwargs):
    """\
    Projected adaptive gradient descent algorithm.

    Parameters:

    x0 : array-like
    Initial parameter array.

    gradient : function(array-like)
    Gradient function. Takes and returns array-like objects with the same shape
    as x0.

    iterations : positive integer
    Number of iterations.

    previous_x : array-like
    Value of Parameter array in previous iteration.

    previous_grad: array-like
    Value of gradient array in previosu iteration.

    Returns:

    x : array-like
    Final parameter array.
    """
    x = x0; i = 0; step_size = np.Inf
    while step_size > min_step and i < max_iters:
        x0 = x
        grad = gradient(x0)
        dgrad = grad-previous_grad
        norm = np.linalg.norm(dgrad)
        if norm == 0.0:
            break
        learning_rate = abs(np.sum((x0-previous_x)*dgrad)/norm**2)
        previous_x = x0
        previous_grad = grad
        x = p(x0 - learning_rate*grad)
        step_size = np.linalg.norm(x-x0)
        i += 1
        
    specs = {
        'iterations' : i,
        'final_step' : step_size,
        'final_gradient' : np.linalg.norm(grad),
        'previous_x' : previous_x,
        'previous_grad' : previous_grad,
        }
    if step_size<=min_step or norm == 0.0:
        specs['stop'] = True
    else:
        specs['stop'] = False
    return x, specs


### Main function ###

algorithms = {
    'gd' : gd,
    'pgd' : pgd,
    'gdm' : gdm,
    'cd' : cd,
    'cdm' : cdm,
    'bgd' : bgd,
    'bgdm' : bgdm,
    'agd' : agd,
    'pagd' : pagd,
}

derivative_type = {
    'gd' : 'full',
    'pgd' : 'full',
    'gdm' : 'full',
    'cd' : 'coordinate',
    'cdm' : 'coordinate',
    'bgd' : 'batch',
    'bgdm' : 'batch',
    'agd' : 'full',
    'pagd' : 'full'
    }

def full(x0, df, f=None, p=None, algorithm='gd', max_iters=100,
         save_gradient=False, save_trajectory=False, save_cost=False,
         save_frequency=1, verbose=0, label=None, **kwargs):
    """\
    Runs GD algorithm and saves/displays computational history.
    """
    algorithm = algorithms[algorithm]

    rounds = math.floor(max_iters/save_frequency)

    steps = np.empty(rounds)
    if save_gradient is True:
        gradient = np.empty(rounds)
    if save_trajectory is True:
        trajectory = np.empty((rounds,)+x0.shape)
    if save_cost is True:
        assert f is not None
        cost = np.empty(rounds)
    
    specs = {}; stop = False; r = 0; x = x0
    while r < rounds and stop is False:
        x, specs = algorithm(x, df, p=p,
                             max_iters = save_frequency, **kwargs, **specs)

        steps[r] = specs['final_step']
        string=f'  iter : {r*save_frequency}, step_size : {steps[r]:0.2e}, '
        if save_gradient is True:
            gradient[r] = specs['final_gradient']
            string += f'gradient_norm : {gradient[r]:0.2e}, '
        if save_trajectory is True:
            trajectory[r] = x
        if save_cost is True:
            cost[r] = f(x)
            string += f'cost : {cost[r]:0.2e}'
        if verbose > 0:
            print(string)
        stop = specs['stop']
        r += 1

    results = {}
    results['rounds'] = r
    results['iterations'] = save_frequency*(r-1)+specs['iterations']
    results['final_step'] = specs['final_step']
    results['final_gradient'] = specs['final_gradient']
    results['steps'] = steps[0:r]
    if save_gradient is True:
        results['gradient'] = gradient[0:r]
    if save_trajectory is True:
        results['trajectory'] = trajectory[0:r]
    if save_cost is True:
        results['cost'] = cost[0:r]

    
    if verbose > 1:
        iters_list = np.arange(0,results['rounds']*save_frequency,
                               save_frequency)
        plt.figure()
        plt.semilogy(iters_list,results['steps'],label='step size')
        if save_gradient is True:
            plt.semilogy(iters_list,results['gradients'],
                         label='gradient size')
        if save_cost is True:
            plt.semilogy(iters_list,results['cost'],label='cost')
        plt.xlabel('iteration')
        plt.title(label)
        plt.legend()
        plt.draw()
        plt.pause(0.2)
        
    return x, results

def main(x0, df, f=None, p=None, algorithm='gd', max_iters=100,
         save_gradient=False, save_trajectory=False, save_cost=False,
         save_frequency=0, verbose=0, label=None, **kwargs):
    """\
    Runs a GD algorithm.
    
    Parameters:

    x0 : numpy array
    Initial parameters to initialize gradient descent method.

    df : function
    Derivative of cost function to be optimized. Given optimization parameters,
    it must return the gradient or partial derivative or batch gradient,
    depending on the algorithm.
    
    algorithm : string
    See the list of methods above.

    f : None or function
    Cost function being optimized (not used in optimization). Necessary if
    save_cost is set to True.

    p : None or function
    Projection function. Necessary if using a projected GD algorithm.

    algorithm : string
    GD algorithm used in optimization.

    save_steps : boolean
    Saves step size and gradient norm during computation, with frequency given
    by save_frequency.

    save_trajectory : boolean
    Saves trajectory during computation, with frequency given by save_frequency.

    save_cost : boolean
    Computes and saves cost during computation, with frequency given by
    save_frequency.

    save_frequency : integer
    Frequency in which data is saved throughout computation. For example, a
    value of 10 results in saving data every 10 steps of the GD algorithm.

    **kwargs : 
    Pass arguments to specific GD method.

    Returns:

    x : numpy array
    Final parameters found by GD algorithm.

    specs : dict
    Dictionary containing save data during computation, as specified by user.
    """
    assert algorithm in algorithms
    if verbose > 0:
        print('# gd.main():')
        print(f'  Algorithm : {algorithm}')

    assert isinstance(save_frequency, int) and save_frequency >= 0
    if save_frequency == 0:
        algorithm = algorithms[algorithm]
        x, specs = algorithm(x0,df,p=p,max_iters=max_iters,**kwargs)
    else:
        x, specs = full(x0,df,f=f,p=p,algorithm=algorithm,max_iters=max_iters,
                        save_gradient=save_gradient,
                        save_trajectory=save_trajectory,save_cost=save_cost,
                        save_frequency=save_frequency,
                        verbose=verbose,label=label,**kwargs)

    if verbose > 0:
        print(f'  Total iterations : {specs["iterations"]}')
        print(f'  Final step size : {specs["final_step"]:0.2e}')
        print(f'  Final gradient norm : {specs["final_gradient"]:0.2e}')
        
    return x, specs

    
### Older gradient descent functions ###

def gradient_descent(x0,df,rate=0.1,max_iters=1000,min_step=1e-6,max_step=1e5,
                     projection=None,trajectory=False,step_history=False,f=None,
                     cost_history=False,feedback=False,plot_history=False):
    """\
    Basic implementation of gradient descent and projected gradient descent.

    The learning rate is fixed. Computation stops after the specified maximum
    number of iterations, or if a minimum/maximum step size conditions are
    reached.

    If a projection function is given, then projected gradient descent is 
    performed.

    If trajectory is True, then the whole trajectory in the algorithm is
    recorded and returned.

    If cost_history is True, then the cost at each step is recorded and given,
    in which case the cost function f must be included.

    If feedback is True, then feedback on the algorithm is printed.

    If plot_history is True, then a plot of the recorded step and/or cost
    histories is passed along (with block=False).

    --- arguments ---
    x0 = initial position (numpy array of any shape and dimension)
    df = gradient of cost function (agreeing with x0)

    rate = learning rate
    max_iters = max number of iterations
    min_step = step size stopping criterion
    max_step = maximum step size stopping criterion (in case of blow-ups)

    projection = projection function

    trajectory = returns trajectory if set to True

    costs = returs cost history if set to True
    f = cost function (include if costs is set to True)

    feedback = set to True to return feedback
    plot_history = set to True to produce history plot
    """
    if feedback is True:
        print("gd.gradient_descent():")
    if f is not None:
        assert callable(f)
        fx0 = f(x0)
        if feedback is True:
            print(f"  initial cost = {fx0:.2e}")
    if projection is not None:
        assert callable(projection)
        project = True
    else:
        project = False
    if trajectory is True:
        xx = [x0.copy()]
    if step_history is True:
        steps = []
    if cost_history is True:
        assert callable(f)
        fx = [fx0]

    x = x0.copy()
    for i in range(max_iters):
        dx = -rate*df(x)
        if project is True:
            x0 = x.copy()
            x = projection(x0+dx)
            dx = x-x0
        else:
            x += dx
        if trajectory is True:
            xx.append(x.copy())
        if cost_history is True:
            fx += [f(x)]
        step_size = np.linalg.norm(dx)
        if step_history is True:
            steps += [step_size]
        if step_size < min_step or step_size > max_step:
            break

    results = dict()
    results['output'] = x
    if trajectory is True:
        results['trajectory'] = xx
    if cost_history is True:
        results['cost_history'] = fx
    if step_history is True:
        results['step_history'] = steps
    if plot_history is True:
        assert step_history is True or cost_history is True
        plt.figure()
        if step_history is True:
            plt.semilogy(steps,label='step size')
        if cost_history is True:
            plt.semilogy(fx,label='cost')
        plt.xlabel('iteration number')
        plt.title('Gradient Descent')
        plt.legend()
        results['figure'] = plt
        plt.show(block=False)
        
    if feedback is True:
        if f is not None:
            print(f"  final cost = {f(x):.2e}")
        
    return results

def coordinate_gradient_descent(x0s,dfs,fs=None,loops=1,rate=0.1,max_iters=1000,
                                min_step=1e-6,max_step=1e5,projection=None,
                                trajectory=False,cost_history=False,
                                feedback=False):
    """\
    Coordinate gradient descent

    --- arguments ---
    x0s = list with initial value for each coordinate. Each coordinate must be a
    numpy array (of any shape and dimensions don't have to agree).
    dfx = list of functions taking list of coordinates and returning gradients
    of each coordinate.
    fs = cost function, taking list of coordinates.
    loops = number of loops of gradient descent around set of coordinates.

    --- kwargs ---
    rate, max_iters, min_step, max_step, projection = same as in gradient
    descent, but these can now be lists (containing one value for each 
    coordinate).
    trajectory, cost_history = saves relevant information if set to True (but 
    only for values after each loop).
    """
    if feedback is True:
        print("gd.coordinate_gradient_descent():")
    coord_num = len(x0s)

    keys = ['rate','max_iters','min_step','max_step','projection']
    values = [rate,max_iters,min_step,max_step,projection]
    vals = dict()
    for i in range(len(keys)):
        if isinstance(values[i],list) is True:
            assert len(values[i]) == coord_num
            vals[keys[i]]=values[i]
        else:
            vals[keys[i]]=[values[i]]*coord_num
            
    xs = copy.deepcopy(x0s)
    for i in range(loops):
        if feedback:
            print(f'  Loop {i}:')
        for j in range(coord_num):
            if feedback:
                print(f'  Coordinate {j}:')
            def df(xj):
                xxs = copy.deepcopy(xs)
                xxs[j] = xj
                return dfs[j](xxs)
            if fs is None:
                f = None
            else:
                def f(xj):
                    xxs = copy.deepcopy(xs)
                    xxs[j]=xj
                    return fs(xxs)
            results = gradient_descent(xs[j],df,f=f,
                                       rate=vals['rate'][j],
                                       max_iters=vals['max_iters'][j],
                                       min_step=vals['min_step'][j],
                                       max_step=vals['max_step'][j],
                                       projection=vals['projection'][j],
                                       plot_history=False,step_history=False,
                                       cost_history=False,feedback=feedback)
            xs[j] = results['output'].copy()

    results = dict()
    results['output'] = xs

    return results

##### Tests #####

def example1():
    print('\n##### TEST #####')
    print('gd.example1():')
    print('Find minimum of f(x)=x^2 using gradient descent with initial '\
          'value 1.')
    x0 = np.array([1.0])
    f = lambda x: x[0]**4
    df = lambda x: 4*x**3
    x, results = full(x0,df,f=f,learning_rate=0.1,algorithm='agd',verbose=2)
    print(f'The minimum of f(x)=x^2 is {x}')
    plt.show()

def example2():
    print('\n##### TEST #####')
    print('gd.example2():')
    print('Find minimum of f(x)=1-<e3,x>^3 for x in sphere using projected '\
          'gradient descent with initial value (sqrt(.99),0,0.1)')
    x0 = np.array([np.sqrt(.99),0,0.1])
    f = lambda x: 1-x[2]**3
    df = lambda x: np.array([0,0,-3*x[2]**2])
    p = lambda x: x/np.linalg.norm(x)
    x, results = full(x0,df,projection=p,f=f,rate=0.01,verbose=2)
    print(f'Solution: {x}')
    plt.show()
    
def example3():
    print('\n##### TEST #####')
    print('gd.example3():')
    print('Find minimum of f(x,y)=x^2+(y-1)^2 using coordinate gradient '\
          'descent with initial value (1,-2).')
    x0s = [np.array([1.0]),np.array([-2.0])]
    fs = lambda x: x[0][0]**2+(x[1][0]-1)**2
    dfs = [lambda x: 2*x[0], lambda x: 2*(x[1]-1)]
    results = coordinate_gradient_descent(x0s,dfs,fs=fs,rate=0.01,feedback=True)
    xs = results['output']
    (x,y) = (xs[0][0],xs[1][0])
    print(f'The minimum of f(x,y)=x^2+(y-1)^2 is ({x},{y}).')

def mds_comparison():
    max_iters = 100
    save_frequency = 10
    
    import misc, distances, mds
    print('\n*** gd.mds_comparison() ***\n')
    
    Y = misc.disk(30,2)
    plt.figure()
    plt.plot(Y[:,0],Y[:,1],'o')
    plt.title('Original data')
    plt.draw()
    plt.pause(0.1)
    
    D = distances.compute(Y)
    
    vis = mds.MDS(D,dim=2,verbose=1)
    vis.initialize_Y()
    vis.optimize(algorithm='gd',max_iters=max_iters,save_cost=True,
                 learning_rate=0.01,from_scratch=True,
                 save_frequency=save_frequency,label='mds, lr=0.01',verbose=2)
    vis.optimize(algorithm='gd',max_iters=max_iters,save_cost=True,
                 learning_rate=0.1,from_scratch=True,
                 save_frequency=save_frequency,label='mds, lr=0.05',verbose=2)
    vis.optimize(algorithm='agd',max_iters=max_iters,save_cost=True,
                 from_scratch=True,save_frequency=save_frequency,verbose=2)

    plt.show()
    
if __name__=="__main__":
    #example1()
    #example2()
    mds_comparison()
