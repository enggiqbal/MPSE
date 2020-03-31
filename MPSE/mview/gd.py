import sys, copy, random, math, numbers, time
import matplotlib.pyplot as plt
import numpy as np

### GRADIENT DESCENT ALGORITHMS (STEP RULES) ###

# Each step rule requires the current position and gradient pair (x,dfx), along
# with parameters specific to each algorithm. The output is the new position x
# after one step, along with a dictionary out, containing the learning rate 'lr'
# used in the step, the total step size 'ndx', a boolean 'stop' if the algorithm
# is not able to produce an update (e.g. due to a zero gradient) and other keys
# necessary to run the algorithm in subsequence iterations.

def fixed(x,dfx,lr=1.0,**kwargs):
    """\
    Fixed learning rate GD scheme.
    """
    dx = -lr*dfx
    ndx = np.linalg.norm(dx)
    x = x+dx
    out = {'lr' : lr,
           'ndx' : ndx,
           'stop' : False}
    return x, out

def bb(x,dfx,x0=0,dfx0=0,**kwargs):
    """\
    Barzilai and Borwein (1988) adaptive GD scheme.
    """
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

def mm(x,dfx,ndx=0,dfx0=0,lr=0.1,theta=np.Inf,alpha=1.0,
              **kwargs):
    """\
    Malitsky and Mishchenko (2019) adaptive GD scheme (algorithm 4).
    """
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

step_rules = {
    'fixed' : fixed,
    'bb' : bb,
    'mm' : mm,
    'adam' : adam
    }

def algorithms(stepping_scheme='fixed',projection=None):
    algorithm0 = step_rules[stepping_scheme]
    if projection is None:
        algorithm = algorithm0
    else:
        def algorithm(x,dfx,**kwargs):
            xx, out = algorithm0(x,dfx,**kwargs)
            pxx = projection(xx)
            out['ndx'] = np.linalg.norm(pxx-x)
            return pxx, out
    return algorithm

### ALGORITHMS ###
    
def single(x0,F,Xi=None,p=None,step_rule='mm',min_cost=None,
           min_grad=None, min_step=None,max_iter=100,max_step=1e4,
           lr=1,verbose=0,plot=False,**kwargs):
    """\
    Gradient descent algorithm, with different options for update rule and 
    stochastic and/or projected variaties.

    Parameters:

    x0 : array
    Initial point.

    F : callable
    Function returning the cost and gradient at a point (either exactly or
    stochastically).
    If Xi is None, the function must be of the form x |-> (f(x),df(x))
    Otherwise, the form is x, xi |-> (f(x,xi),df(x,xi)).

    Xi : None or callable
    If Xi is None, then F is exact.
    Otherwise, Xi() produces stochastic parameters xi.

    p : None or callable
    If callable, then updates are projected (constraint optimization).
    It takes the form x -> p(x).
    """
    if Xi is None:
        stochastic = False
    else:
        stochastic = True
    if p is None:
        constraint = False
    else:
        constraint = True
    assert step_rule in step_rules
    algorithm = algorithms(step_rule,p)
    
    if verbose > 0:
        print('- gd.single(): ')
        print('  computation parameters:')
        print(f'    constraint : {constraint}')
        print(f'    update rule : {step_rule}')
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

    normalization = math.sqrt(np.size(x0))
    x = x0
    for i in range(5):
        x0 = x.copy()
        if stochastic is False:
            fx0, dfx0 = F(x0)
        else:
            fx0, dfx0 = F(x0,Xi())
        dx = -lr*dfx0
        ndx = np.linalg.norm(dx)
        x = x0 + dx
    success = True
    conclusion = 'maximum number of iterations reached'
    kwargs['ndx'] = ndx
    kwargs['lr'] = lr
    if verbose > 1:
        print('  progress:')
    for i in range(max_iter):
        if stochastic is False:
            fx, dfx = F(x) #cost and gradient evaluated at x
        else:
            xi = Xi()
            fx, dfx = F(x,xi)
        grads[i] = np.linalg.norm(dfx)/normalization #rms of gradient
        costs[i] = fx
        if fx < min_cost:
            conclusion = 'minimum cost reached'
            costs = costs[0:i+1]
            grads = grads[0:i+1]
            lrs = lrs[0:i]
            steps = steps[0:i]
            break
        if grads[i] < min_grad:
            conclusion = 'minimum gradient size reached'
            costs = costs[0:i+1]
            grads = grads[0:i+1]
            lrs = lrs[0:i]
            steps = steps[0:i]
            break
        if stochastic is True and step_rule in ['bb','mm']:
            fx0, dfx0 = F(x0,xi)
            kwargs['dfx0'] = dfx0
        x0 = x
        x, kwargs = algorithm(x,dfx,**kwargs)
        if kwargs['stop'] == True:
            conclusion = 'update rule'
            break
        lrs[i] = kwargs['lr']
        steps[i] = kwargs['ndx']/normalization #rms of step size
        if steps[i] < min_step:
            conclusion = 'minimum step reached reached'
            costs = costs[0:i+1]
            grads = grads[0:i+1]
            lrs = lrs[0:i+1]
            steps = steps[0:i+1]
            break
        elif steps[i] > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            costs = costs[0:i+1]
            grads = grads[0:i+1]
            lrs = lrs[0:i+1]
            steps = steps[0:i+1]
            break
        if verbose > 1:
            print(f'    {i:>4}/{max_iter} : step = {steps[i]:0.2e}, grad = {grads[i]:0.2e}, cost = {costs[i]:0.2e}, lr = {lrs[i]:0.2e}',
                  flush=True, end="\r")
    
    tf = time.time()

    if plot is True:
        fig, ax = plt.subplots()
        ax.semilogy(costs,label='cost',linewidth=3)
        ax.semilogy(grads,label='gradient size',linestyle='--')
        ax.semilogy(lrs,label='learning rate',linestyle='--')
        ax.semilogy(steps,label='step size',linestyle='--')
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
        
    if verbose > 1:
        print('  results:')
        print(f'    conclusion : {conclusion}')
        print(f'    total iterations : {i}')
        print(f'    final cost : {costs[-1]:0.2e}')
        print(f'    final gradient size : {grads[-1]:0.2e}')        
        print(f'    final learning rate : {lrs[-1]:0.2e}')
        print(f'    final step size : {steps[-1]:0.2e}')
        print(f'    time : {tf-t0:0.2e} [sec]')
    return x, outputs

def multiple(X0,F,Xi=None,p=None,step_rule='fixed',min_cost=None,
             min_grad=None, min_step=None,max_iter=100,max_step=1e4,
             lr=1,verbose=0,plot=False,**kwargs):
    """\
    Gradient descent algorithms.
    """
    assert isinstance(X0,list); K = len(X0)

    if Xi is None:
        stochastic = False
    else:
        stochastic = True
        
    if isinstance(p,list):
        assert len(p) == K
    else:
        p = [p]*K
    if isinstance(step_rule,list):
        assert len(step_rule) == K
    else:
        step_rule = [step_rule]*K
    if isinstance(lr,list):
        assert len(lr) == K
    else:
        lr = [lr]*K

    constraint = []
    algorithm = []
    for k in range(K):
        algorithm.append(algorithms(step_rule[k],p[k]))
        if p[k] is None:
            constraint.append(False)
        else:
            constraint.append(True)
    if verbose > 0:
        print('- gd.multiple(): ')
        print('  computation parameters:')
        #print(f'    constraint : {constraint}')
        print(f'    update rule : {step_rule}')
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

    normalization = [math.sqrt(np.size(a)) for a in X0]
    if stochastic is False:
        fX0, dfX0 = F(X0)
    else:
        print(Xi)
        fX0, dfX0 = F(X0,Xi())
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
    if verbose > 1:
        print('  progress:')
    for i in range(max_iter):
        if stochastic is False:
            fX, dfX = F(X)
        else:
            xi = Xi()
            fX, dfX = F(X,xi)
        grads[i] = [np.linalg.norm(a)/b for a, b in zip(dfX,normalization)]
        costs[i] = fX
        if fX < min_cost:
            conclusion = 'minimum cost reached'
            break
        if max(grads[i]) < min_grad:
            conclusion = 'minimum gradient size reached'
            break
        if stochastic is True:
            fX0, dfX0 = F(X0,xi)
        for k in range(K):
            if stochastic is True:
                KWARGS[k]['dfx0'] = dfX0[k]
            X[k], KWARGS[k] = algorithm[k](X[k],dfX[k],**KWARGS[k])
            if KWARGS[k]['stop'] == True:
                conclusion = 'update rule'
                break
            lrs[i,k] = KWARGS[k]['lr']
            steps[i,k] = KWARGS[k]['ndx']/normalization[k]
        if max(steps[i]) < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif max(steps[i]) > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 1:
            #sys.stdout.write("\033[K")
            print(f'    {i:>4}/{max_iter} : step = {np.max(steps[i]):0.2e}, '+\
                  f'grad = {np.max(grads[i]):0.2e}, cost = {costs[i]:0.2e}, '+\
                  f'lr = {np.max(lrs[i]):0.2e}',
                  flush=True, end="\r")
            #sys.stdout.write("\033[F")

    if verbose > 1:
        sys.stdout.write("\033[K")
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    tf = time.time()

    costs = costs[0:i]
    grads = grads[0:i]
    lrs = lrs[0:i]
    steps = steps[0:i]

    if plot is True:
        fig, axes = plt.subplots(1,1+K,figsize=(15,5))
        axes[0].semilogy(costs,linewidth=3)
        axes[0].set_title('cost')
        for k in range(K):
            axes[k+1].semilogy(grads[:,k],label='gradient size',linestyle='--')
            axes[k+1].semilogy(lrs[:,k],label='learning rate',linestyle='--')
            axes[k+1].semilogy(steps[:,k],label='step size',linestyle='--')
            axes[k+1].set_title(f'coordinate {k}')
            axes[k+1].legend()
            axes[k+1].set_xlabel('iterations')
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
        
    if verbose > 1:
        print('  results:')
        print(f'    conclusion : {conclusion}')
        print(f'    total iterations : {i}')
        print(f'    final cost : {costs[-1]:0.2e}')
        print(f'    time : {tf-t0:0.2e} [sec]')
    return X, outputs


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
