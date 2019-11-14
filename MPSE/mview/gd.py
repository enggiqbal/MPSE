import copy, random, math, numbers
import matplotlib.pyplot as plt
import numpy as np
import sys
### GD step functions ###

def gd(x0,F,p=None,min_step=1e-6,max_iters=100,max_step=1e4,lr=0.1,
       verbose=0,**kwargs):
    """\
    Gradient descent algorithm.

    Parameters:

    x0 : array-like
    Initial parameter array.

    F : function(array-like)
    Function returning the tuple containing the cost and gradient at the 
    parameters of interest. Given an array x (same shape as x0), it returns
    (f(x0),df(x0)), where f is the cost function and df is the gradient 
    function.

    min_step : number > 0
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped and declared successful.

    max_iters : integer > 0
    Maximum number of iterations.

    max_step : number > 0
    Maximum step size allowed. If step size is larger, the computation is 
    stopped and declared failed.

    lr : number > 0
    Learning rate.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation, including cost history.
    """
    if verbose > 0:
        print('. gd.gd(): ')
        print('  min_step : {min_step:0.2e}')
        print('  max_iters :', max_iters)
        print('  max_step : {max_step:0.2e}')
        print('  lr : {lr:0.2e}')
        sys.stdout.flush()
        
    if p is None:
        alg = lambda x, dx: x-lr*dx
    else:
        alg = lambda x, dx: p(x-lr*dx)

    cost = np.empty(max_iters)
    steps = np.empty(max_iters)
    grads = np.empty(max_iters)
    
    x = x0.copy(); i = 0; step_size = (min_step+max_step)/2;
    while min_step < step_size < max_step and i < max_iters:
        x0 = x
        cost[i], grad = F(x0)
        x = alg(x0,grad)
        step_size = np.linalg.norm(x-x0); steps[i] = step_size
        grad_size = np.linalg.norm(grad); grads[i] = grad_size
        if verbose > 1:
            print('  {i:>4} : step = {steps[i]:0.2e}, grad = ' +
                  f'{grads[i]:0.2e}, cost = {cost[i]:0.2e}')
        i += 1
        
    specs = {
        'cost' : cost[0:i],
        'steps' : steps[0:i],
        'grads' : grads[0:i],
        'iterations' : i,
        'x_prev' : x0,
        'dx_prev' : grad,
        'minimum_reached' : step_size <= min_step,
        'unstable' : step_size > max_step
        }
    if verbose > 0:
        print('  total iterations :',i)
        print('  final step size : {step_size:0.2e}')
        print('  final gradient size: {np.linalg.norm(grad):0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
        sys.stdout.flush()
    return x, specs

def agd(x0,F,p=None,min_step=1e-6,max_iters=100,max_step=1e4,x_prev=0,dx_prev=0,
       verbose=0,**kwargs):
    """\
    Adaptive gradient descent algorithm.

    Parameters:

    x0 : array-like
    Initial parameter array.

    F : function(array-like)
    Function returning the tuple containing the cost and gradient at the 
    parameters of interest. Given an array x (same shape as x0), it returns
    (f(x0),df(x0)), where f is the cost function and df is the gradient 
    function.

    min_step : number > 0
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped and declared successful.

    max_iters : integer > 0
    Maximum number of iterations.

    max_step : number > 0
    Maximum step size allowed. If step size is larger, the computation is 
    stopped and declared failed.

    x_prev : array-like
    Parameter array obtained one previous step before initial parameter array
    (optional). Used to more accurately define initial learning rate. Include if
    agd is used after some other gd algorithm.

    dx_prev : array-like
    gradient array obtained one previous step before initial parameter array.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation, including cost history.
    """
    if verbose > 0:
        print('. gd.agd(): ')
        print('  min_step : {min_step:0.2e}')
        print('  max_iters :', max_iters)
        print('  max_step : {max_step:0.2e}')
        
    if p is None:
        alg = lambda x, dx, lr: x-lr*dx
    else:
        alg = lambda x, dx, lr: p(x-lr*dx)

    cost = np.empty(max_iters)
    steps = np.empty(max_iters)
    grads = np.empty(max_iters)
    
    x = x0.copy(); x0 = x_prev; grad0=dx_prev; i = 0
    step_size = (min_step+max_step)/2;
    while min_step < step_size < max_step and i < max_iters:
        cost[i], grad = F(x)
        dgrad = grad-grad0
        norm = np.linalg.norm(dgrad)
        if norm == 0.0:
            stop = True
            break
        else:
            stop = False
            lr = abs(np.sum((x-x0)*dgrad)/norm**2)
            x0 = x; grad0 = grad
            x = alg(x,grad,lr)
        step_size = np.linalg.norm(x-x0); steps[i] = step_size
        grad_size = np.linalg.norm(grad); grads[i] = grad_size
        if verbose > 1:
            print('  {i:>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e},'+
                  f' cost = {cost[i]:0.2e}')
        i += 1
        
    specs = {
        'cost' : cost[0:i],
        'steps' : steps[0:i],
        'grads' : grads[0:i],
        'iterations' : i,
        'x_prev' : x0,
        'dx_prev' : grad,
        'minimum_reached' : step_size <= min_step,
        'unstable' : step_size > max_step
        }
    if verbose > 0:
        print('  total iterations :',i)
        print('  final step size : {step_size:0.2e}')
        print('  final gradient size: {np.linalg.norm(grad):0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
        sys.stdout.flush()
    return x, specs

def cgd(X0,F,p=None,max_iters=200,min_step=1e-15,max_step=1e4,lr=0.1,
        verbose=0,**kwargs):
    """\
    Coordinate gradient descent algorithm.

    Parameters:

    X0 : list of array-like objects
    List of Initial parameter arrays.

    F : function(array-like)
    Function returning the tuple containing the cost and list of gradients at 
    the list of parameters of interest. Given a list of arrays X, it returns
    (f(X),df(X)), where f is the cost function and df is the gradient 
    function.

    min_step : number > 0
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped and declared successful.

    max_iters : integer > 0
    Maximum number of iterations.

    max_step : number > 0
    Maximum step size allowed. If step size is larger, the computation is 
    stopped and declared failed.

    lr : number > 0
    Learning rate.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation, including cost history.
    """
    if verbose > 0:
        print('+ gd.cgd():')
        print('  min_step : {min_step:0.2e}')
        print('  max_iters :',max_iters)
        print('  max_step : {max_step:0.2e}')
        print('  lr : ',lr)
        
    assert isinstance(X0,list); K = len(X0)
    if isinstance(lr,numbers.Number):
        lr = [lr]*K
    else:
        assert isinstance(lr,list) or isinstance(lr,np.ndarray)
    alg = []
    for k in range(K):
        if p is None or p[k] is None:
            alg.append(lambda x, dx: x-lr[k]*dx)
        else:
            alg.append(lambda x, dx: p[k](x-lr[k]*dx))
        
    cost = np.empty(max_iters)
    steps = np.empty(max_iters)
    grads = np.empty(max_iters)
    
    X = X0; i = 0;  step_size = (min_step+max_step)/2; specs=[{}]*K
    while min_step < step_size < max_step and i < max_iters:
        X0 = X.copy()
        cost[i], GRAD = F(X0)
        step_size = 0; grad_size = 0
        for k in range(K):
            X[k] = alg[k](X0[k],GRAD[k])
            step_size += np.linalg.norm(X[k]-X0[k])
            grad_size += np.linalg.norm(GRAD[k])
        steps[i] = step_size; grads[i] = grad_size
        print('  {i>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e}, '+
              f'cost = {cost[i]:0.2e}')
        sys.stdout.flush()
        i += 1

    specs = {
        'cost' : cost[0:i],
        'steps' : steps[0:i],
        'grads' : grads[0:i],
        'iterations' : i,
        'x_prev' : X0,
        'dx_prev' : GRAD,
        'minimum_reached' : step_size <= min_step,
        'unstable' : step_size > max_step
        }
    if verbose > 0:
        print('  total iterations :',i)
        print('  final step size : {step_size:0.2e}')
        print('  final gradient size: {grad_size:0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
        sys.stdout.flush()
    return X, specs

def cagd(X0,F,p=None,max_iters=200,min_step=1e-15,max_step=1e4,X_prev=0,
         dX_prev=0,verbose=0,**kwargs):
    """\
    Coordinate gradient descent algorithm.

    Parameters:

    X0 : list of array-like objects
    List of Initial parameter arrays.

    F : function(array-like)
    Function returning the tuple containing the cost and list of gradients at 
    the list of parameters of interest. Given a list of arrays X, it returns
    (f(X),df(X)), where f is the cost function and df is the gradient 
    function.

    min_step : number > 0
    Minimum step size required to continue computation. If a step size goes
    below this threshold, the computation is stopped and declared successful.

    max_iters : integer > 0
    Maximum number of iterations.

    max_step : number > 0
    Maximum step size allowed. If step size is larger, the computation is 
    stopped and declared failed.

    X_prev : list of array-like objects, same shape as X0
    List of parameter arrays obtained one previous step before initial parameter
    array (optional). Used to more accurately define initial learning rate. 
    Include if agd is used after some other gd algorithm.

    dX_prev : list of array-like objects, same shape as X0
    List of gradient arrays obtained one previous step before initial parameter
    array.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation, including cost history.
    """
    if verbose > 0:
        print('+ gd.cagd():')
        print('  min_step : {min_step:0.2e}')
        print('  max_iters :',max_iters)
        print('  max_step : {max_step:0.2e}')
        sys.stdout.flush()
        
    assert isinstance(X0,list); K = len(X0)
    alg = []
    for k in range(K):
        if p is None or p[k] is None:
            alg.append(lambda x, dx, lr: x-lr*dx)
        else:
            alg.append(lambda x, dx, lr: p[k](x-lr*dx))

    if X_prev == 0.0:
        X_prev = [0]*K
    if dX_prev == 0.0:
        dX_prev = [0]*K
        
    cost = np.empty(max_iters)
    steps = np.empty(max_iters)
    grads = np.empty(max_iters)
    
    X = X0.copy(); X0 = X_prev; dX0 = dX_prev; i = 0;
    step_size = (min_step+max_step)/2; stop=0
    while min_step < step_size < max_step and i < max_iters and stop<K:
        sys.stdout.flush()        

        cost[i], dX = F(X)
        step_size = 0; grad_size = 0; stop = 0
        for k in range(K):
            dgrad = dX[k]-dX0[k]
            norm = np.linalg.norm(dgrad)
            if norm == 0.0:
                stop += 1
            else:
                lr = abs(np.sum((X[k]-X0[k])*dgrad)/norm**2)
                X0[k] = X[k]; dX0[k] = dX[k]
                X[k] = alg[k](X[k],dX[k],lr)
            step_size += np.linalg.norm(X[k]-X0[k])
            grad_size += np.linalg.norm(dX[k])
        steps[i] = step_size; grads[i] = grad_size
        print('  {i:>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e}, '+
              f'cost = {cost[i]:0.2e}')
        i += 1

    specs = {
        'cost' : cost[0:i],
        'steps' : steps[0:i],
        'grads' : grads[0:i],
        'iterations' : i,
        'x_prev' : X0,
        'dx_prev' : dX0,
        'minimum_reached' : step_size <= min_step,
        'unstable' : step_size > max_step
        }
    if verbose > 0:
        print('  total iterations :',i)
        print('  final step size : {step_size:0.2e}')
        print('  final gradient size: {grad_size:0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
        sys.stdout.flush()
    return X, specs

def mgd(x0,F,lr=0.1,attempts=10,reduce_factor=10,verbose=0,**kwargs):
    """\
    Scheme to run gradient descent or coordinate gradient descent multiple times
    until a satisfactory result is obtained. This is done by reducing learning
    rate by a reduction factor if computation fails, until success.

    Parameters:

    x0 : array-like object or list of array-like objects
    Initial parameter array or list of initial parameter array.

    F : function(array-like)
    Function returning the tuple containing the cost and gradient(s) at the 
    parameters of interest. Given an array or list of arrays x (same shape as 
    x0), it returns (f(x),df(x)), where f is the cost function and df is the 
    gradient(s) function.

    lr : number > 0
    Learning rate. If running coordinate gradient descent, then lr can be a list
    containing the learning rates for each coordinate.

    attemps : int > 0
    Number of attempts in reaching a stable solution.

    reduce_factor : number > 0
    Reduce learning rate by this factor after each failure.

    Returns:

    x : numpy array
    Final parameter array.

    specs : dictionary
    Dictionary with information about computation, including cost history.
    """
    if verbose > 0:
        print('. gd.mgd():')

    if isinstance(x0,np.ndarray):
        alg = gd
        assert isinstance(lr,numbers.Number)
    elif isinstance(x0,list):
        alg= cgd
        if isinstance(lr,numbers.Number):
            lr = np.array([lr]*len(x0))
        elif isinstance(lr,list):
            lr = np.array(lr)
        else:
            assert isinstance(lr,np.ndarray)
    else:
        sys.exit('Initial parameters must be a numpy array or a list')

    attempt = 0
    while attempt < attempts:
        if verbose > 0:
            print('  attempt ',attempt,':')
        x, specs = alg(x0.copy(),F,lr=lr,verbose=verbose,**kwargs)
        attempt += 1
        if specs['unstable']:
            lr /= reduce_factor
        else:
            specs['final_lr'] = lr
            break
        
    if verbose > 0:
        print('  number of attempts :',attempt)

    return x, specs

### TESTS ###

def mds_comparison(N=100,**kwargs):
    
    import misc, distances, mds
    print('\n*** gd2.mds_comparison(): ***\n')
    
    Y = misc.disk(N,3)
    D = distances.compute(Y)
    
    vis = mds.MDS(D,dim=3,verbose=1)
    vis.initialize()

    label = 'gd, lr=0.005'
    vis.optimize(algorithm='gd',lr=0.005,label=label,
                 verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()

    label = 'gdm, lr=0.05'
    vis.optimize(algorithm='gdm',lr=0.05,label=label,
                 verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()

    label = 'agd'
    vis.optimize(algorithm='agd',label=label,
                 verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()
    
    plt.show()

def mmds_comparison(N=100,**kwargs):
    
    import misc, distances, perspective,  multiview
    print('\n*** gd2.mmds_comparison(): ***\n')
    
    X = misc.disk(N,3)
    persp = perspective.Persp()
    persp.fix_Q(number=3,special='standard')
    Y = persp.compute_Y(X)
    D = distances.compute(Y)
    
    vis = multiview.Multiview(D,persp=persp,verbose=1)
    vis.setup_visualization()
    vis.initialize_X(method='mds')

    label = 'gd, lr=0.001'
    vis.optimize_all(algorithm='gd',lr=0.001,label=label,
                     verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()

    label = 'gdm, lr=0.001'
    vis.optimize_all(algorithm='gdm',lr=0.001,label=label,
                     verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()

    label = 'agd'
    vis.optimize_all(algorithm='agd',label=label,
                     verbose=1,**kwargs)
    vis.figureH(label)
    vis.forget()
    
    plt.show()
    
if __name__=="__main__":
    #mds_comparison(N=100,max_iters=100)
    mmds_comparison(N=100,max_iters=100)
