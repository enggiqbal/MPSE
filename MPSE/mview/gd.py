import copy, random, math, numbers, time
import matplotlib.pyplot as plt
import numpy as np

### UPDATE RULES ###

def fixed(x,dfx,lr=1.0,**kwargs):
    dx = -lr*dfx
    ndx = np.linalg.norm(dx)
    x = x+dx
    out = {'lr' : lr,
           'ndx' : ndx,
           'stop' : False}
    return x, out

def adaptive1(x,dfx,x0=0,dfx0=0,**kwargs):
    ddfx = dfx-dfx0
    nddfx = np.linalg.norm(ddfx)
    if nddfx == 0.0:
        out = {
            'stop' : True
            }
    else:
        stop = False
        dx = x-x0
        ndx = np.linalg.norm(dx)
        lr = abs(np.sum(dx*ddfx))/nddfx**2
        x0 = x
        dfx0 = dfx
        dx = -lr*dfx
        x = x+dx
        out = {
            'lr' : lr,
            'ndx' : ndx,
            'x0' : x0,
            'dfx0' : dfx0,
            'stop' : False
        }
    return x, out

def adaptive2(x,dfx,ndx=0,dfx0=0,lr=0.1,theta=np.Inf,alpha=1.0,
              **kwargs):
    if ndx == 0:
        out = {
            'stop' : True
            }
    else:
        nddfx = np.linalg.norm(dfx-dfx0)
        L = nddfx/ndx
        lr0 = lr
        lr = min(math.sqrt(1+theta)*lr,1/(alpha*L))
        theta = lr/lr0
        dx = -lr*dfx
        ndx = np.linalg.norm(dx)
        x = x + dx
        dfx0 = dfx
        out = {
            'ndx' : ndx,
            'dfx0' : dfx0,
            'lr' : lr,
            'theta' : theta,
            'alpha' : alpha,
            'stop' : False
            }
    return x, out

def adam(x,dfx,ndx=0,dfx0=0,lr=0.1,m=0,v=0,i=0,**kwargs):
    """\
    ADAM gradient descent implementation.
    See paper ADAM

    x0 : array-like
    Initial parameter array.

    F : function(array-like)
    Function returning (f(x),df(x)) at a given parameter X, where f is
    the cost function (or an approximation to it) and df is the gradient
    function.
    """
    beta1=0.9
    beta2=0.999
    epsilon=1e-8

    m = beta1*m+(1-beta1)*dfx
    v = beta2*v+(1-beta2)*dfx**2
    mc = m/(1-beta1**(i+1)) #corrected first moment estimate
    vc = v/(1-beta2**(i+1)) #corrected second raw moment estimate
    dx = -lr*mc/(np.sqrt(vc)+epsilon)
    ndx = np.linalg.norm(dx)
    x = x + dx
    out = {'lr' : lr,
           'ndx' : ndx,
           'm' : m,
           'v' : v,
           'i' : i+1,
           'stop' : False}
    return x, out

update_rules = {
    'fixed' : fixed,
    'adaptive1' : adaptive1,
    'adaptive2' : adaptive2,
    'adam' : adam
    }

def add_constraint(algorithm, p, x, dfx, **kwargs):
    """\
    Updates x and out using projection p
    """
    xx, out = algorithm(x,dfx,**kwargs)
    pxx = p(xx)
    out['ndx'] = np.linalg.norm(pxx-x)
    return pxx, out

### ALGORITHMS ###
    
def single(x0,F,p=None,update_rule='fixed',min_cost=None,
           min_grad=None, min_step=None,max_iter=100,max_step=1e4,
           lr=0.1,verbose=0,plot=False,**kwargs):
    """\
    Gradient descent algorithms.
    """
    assert update_rule in update_rules
    if p is None:
        constraint = False
        algorithm = update_rules[update_rule]
    else:
        constraint = True
        alg = update_rules[update_rule]
        algorithm = lambda x, dfx, **kwargs : \
              add_constraint(alg,p,x,dfx,**kwargs)
    if verbose > 0:
        print('- gd.single(): ')
        print('  computation parameters:')
        print(f'    constraint : {constraint}')
        print(f'    update rule : {update_rule}')
        if min_cost is not None:
            print(f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(f'    min_step : {min_step:0.2e}')
        print(f'    max_iter : {max_iter}')
        print(f'    max_step : {max_step:0.2e}')
    if min_cost is None:
        min_cost = -np.Inf
    if min_grad is None:
        min_grad = -np.Inf
    if min_step is None:
        min_step = -np.Inf

    costs = np.empty(max_iter)
    grads = np.empty(max_iter)
    steps = np.empty(max_iter)
    lrs = np.empty(max_iter)

    t0 = time.time()

    fx0, dfx0 = F(x0)
    dx = -lr*dfx0
    ndx = np.linalg.norm(dx)
    x = x0 + dx
    success = True
    conclusion = 'maximum number of iterations reached'
    kwargs['ndx'] = ndx
    kwargs['lr'] = lr
    for i in range(max_iter):
        fx, dfx = F(x); ndfx = np.linalg.norm(dfx)
        costs[i] = fx; grads[i] = ndfx
        if fx < min_cost:
            conclusion = 'minimum cost reached'
            break
        if ndfx < min_grad:
            conclusion = 'minimum gradient size reached'
            break
        x, kwargs = algorithm(x,dfx,**kwargs)
        if kwargs['stop'] == True:
            conclusion = 'update rule'
            break
        lrs[i] = kwargs['lr']
        steps[i] = kwargs['ndx']
        if kwargs['ndx'] < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif kwargs['ndx'] > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 1:
            print(f'  {i:>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e},'+
                  f' cost = {costs[i]:0.2e}, lr = {lrs[i]:0.2e}',
                  flush=True, end="\r")
    
    tf = time.time()

    costs = costs[0:i]
    grads = grads[0:i]
    lrs = lrs[0:i]
    steps = steps[0:i]

    if plot is True:
        fig, ax = plt.subplots()
        ax.semilogy(costs,label='costs')
        ax.semilogy(grads,label='gradient size')
        ax.semilogy(lrs,label='learning rate')
        ax.semilogy(steps,label='step size')
        ax.legend()
        ax.set_xlabel('iterations')
        plt.draw()
        plt.pause(0.1)
    
    outputs = {
        'costs' : costs,
        'steps' : steps,
        'grads' : grads,
        'lrs' : lrs,
        'iterations' : i,
        'success' : success,
        'conclusion' : conclusion,
        'time' : tf-t0
        }
        
    if verbose > 2:
        print('  results:')
        print(f'    conclusion : {conclusion}')
        print(f'    total iterations : {i}')
        print(f'    final cost : {costs[-1]:0.2e}')
        print(f'    final gradient size : {grads[-1]:0.2e}')        
        print(f'    final learning rate : {lrs[-1]:0.2e}')
        print(f'    final step size : {steps[-1]:0.2e}')
        print(f'    time : {tf-t0:0.2e} [sec]')
    return x, outputs

def multiple(X0,F,p=None,update_rule='fixed',min_cost=None,
             min_grad=None, min_step=None,max_iter=100,max_step=1e4,
             lr=0.1,verbose=0,plot=False,**kwargs):
    """\
    Gradient descent algorithms.
    """
    assert isinstance(X0,list); K = len(X0)
    if isinstance(p,list):
        assert len(p) == K
    else:
        p = [p]*K
    if isinstance(update_rule,list):
        assert len(update_rule) == K
    else:
        update_rule = [update_rule]*K
    if isinstance(lr,list):
        assert len(lr) == K
    else:
        lr = [lr]*K

    constraint = []
    algorithm = []
    for k in range(K):
        assert update_rule[k] in update_rules
        alg = update_rules[update_rule[k]]
        if p[k] is None:
            constraint.append(False)
            algorithm.append(alg)
        else:
            constraint.append(True)
            algorithm.append(lambda x, dfx, **kwargs: \
                             add_constraint(alg,p[k],x,dfx,**kwargs))
    if verbose > 0:
        print('- gd.multiple(): ')
        print('  computation parameters:')
        #print(f'    constraint : {constraint}')
        print(f'    update rule : {update_rule}')
        if min_cost is not None:
            print(f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(f'    min_step : {min_step:0.2e}')
        print(f'    max_iter : {max_iter}')
        print(f'    max_step : {max_step:0.2e}')
    if min_cost is None:
        min_cost = -np.Inf
    if min_grad is None:
        min_grad = -np.Inf
    if min_step is None:
        min_step = -np.Inf

    costs = np.empty(max_iter)
    grads = np.empty((max_iter,K))
    steps = np.empty((max_iter,K))
    lrs = np.empty((max_iter,K))

    t0 = time.time()

    fX0, dfX0 = F(X0)
    dX = [-a*b for a,b in zip(lr,dfX0)]
    ndX = [np.linalg.norm(a) for a in dX]
    X = [a+b for a,b in zip(X0,dX)]
    success = True
    conclusion = 'maximum number of iterations reached'
    KWARGS = []
    for k in range(K):
        KWARGS.append(copy.deepcopy(kwargs))
    for i in range(K):
        KWARGS[i]['ndx'] = ndX[i]
        KWARGS[i]['lr'] = lr[i]
    for i in range(max_iter):
        fX, dfX = F(X); ndfX = [np.linalg.norm(a) for a in dfX]
        costs[i] = fX; grads[i] = ndfX
        if fX < min_cost:
            conclusion = 'minimum cost reached'
            break
        if max(ndfX) < min_grad:
            conclusion = 'minimum gradient size reached'
            break
        for k in range(K):
            X[k], KWARGS[k] = algorithm[k](X[k],dfX[k],**KWARGS[k])
            if KWARGS[k]['stop'] == True:
                conclusion = 'update rule'
                break
            lrs[i,k] = KWARGS[k]['lr']
            steps[i,k] = KWARGS[k]['ndx']
       # if max(kwargs['ndx']) < min_step:
        #    conclusion = 'minimum step reached reached'
         #   break
        #elif max(kwargs['ndx']) > max_step:
         #   success = False
          #  conclusion = 'maximum step size reached (unstable)'
           # break
        if verbose > 1:
            print(f'  {i:>4} : step = {steps[i,0]:0.2e}, grad = {grads[i,0]:0.2e}, cost = {costs[i]:0.2e}, lr = {lrs[i,0]:0.2e}', flush=True, end="\r")
    
    tf = time.time()

    costs = costs[0:i]
    grads = grads[0:i]
    lrs = lrs[0:i]
    steps = steps[0:i]

    if plot is True:
        fig, ax = plt.subplots()
        ax.semilogy(costs,label='costs')
        ax.semilogy(grads,label='gradient size')
        ax.semilogy(lrs,label='learning rate')
        ax.semilogy(steps,label='step size')
        ax.legend()
        ax.set_xlabel('iterations')
        plt.draw()
        plt.pause(0.1)
    
    outputs = {
        'costs' : costs,
        'steps' : steps,
        'grads' : grads,
        'lrs' : lrs,
        'iterations' : i,
        'success' : success,
        'conclusion' : conclusion,
        'time' : tf-t0
        }
        
    if verbose > 2:
        print('  results:')
        print(f'    conclusion : {conclusion}')
        print(f'    total iterations : {i}')
        print(f'    final cost : {costs[-1]:0.2e}')
        print(f'    final gradient size : {grads[-1]:0.2e}')        
        print(f'    final learning rate : {lrs[-1]:0.2e}')
        print(f'    final step size : {steps[-1]:0.2e}')
        print(f'    time : {tf-t0:0.2e} [sec]')
    return X, outputs

def cgd(X0,F,p=None,max_iters=200,min_step=1e-15,max_step=1e4,lr=0.1,
        verbose=1,**kwargs):
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
        print(f'  min_step : {min_step:0.2e}')
        print('  max_iters :',max_iters)
        print(f'  max_step : {max_step:0.2e}')
        print(f'  lr : ',lr)
        
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
        if verbose > 1:
            print(f'  {i>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e}, '+
                  f'cost = {cost[i]:0.2e}', flush=True)
        i += 1

    specs = {
        'cost' : cost[0:i],
        'steps' : steps[0:i],
        'grads' : grads[0:i],
        'iterations' : i,
        'X_prev' : X0,
        'dX_prev' : GRAD,
        'minimum_reached' : step_size <= min_step,
        'unstable' : step_size > max_step
        }
    if verbose > 0:
        print('  total iterations :',i)
        print(f'  final step size : {step_size:0.2e}')
        print(f'  final gradient size: {grad_size:0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
            
    return X, specs

def cagd(X0,F,p=None,max_iters=200,min_step=1e-15,max_step=1e4,X_prev=0,
         dX_prev=0,verbose=1,**kwargs):
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
        print(f'  min_step : {min_step:0.2e}')
        print('  max_iters :',max_iters)
        print(f'  max_step : {max_step:0.2e}')
        
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
        if verbose > 1:
            print(f'  {i:>4} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e},'
                  +f' cost = {cost[i]:0.2e}', flush=True)
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
        print(f'  final step size : {step_size:0.2e}')
        print(f'  final gradient size: {grad_size:0.2e}')
        if specs['minimum_reached'] is True:
            print('  LOCAL MINIMUM REACHED')
        if specs['unstable'] is True:
            print('  UNSTABLE')
            
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
        print('  number of attempts :',attempt, flush=True)

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
