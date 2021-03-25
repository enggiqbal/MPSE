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

def fixed(x,dfx,lr=1.0,p=None,**kwargs):
    """\
    Fixed learning rate GD scheme.
    """
    dx = -lr*dfx #step against gradient
    ndx = np.linalg.norm(dx) #step size against gradient
    y = x+dx #position after step against gradient (before projection)
    if p is None:
        x = y #position after step and projecion
        step = ndx #step size after step and projection
    else:
        x_new = p(y) #position after step and projection
        step = np.linalg.norm(x_new-x) #step size after step and projection
        x = x_new
    out = {'lr' : lr,
           'ndx' : ndx,
           'y' : y,
           'step' : step,
           'df0x0' : dfx,
           'stop' : False}
    return x, out

def bb(x,dfx,x0=0,dfx0=0,p=None,y=None,**kwargs):
    """\
    Barzilai and Borwein (1988) adaptive GD scheme.

    x : current position
    dfx : current gradient
    x0 : previous position
    dfx0 : gradient at previous position (using same stochastic approx)
    p : projection function
    y : current position before projection
    """
    x_initial = x
    ddfx = dfx-dfx0
    nddfx = np.linalg.norm(ddfx)
    if nddfx == 0.0:
        out = {
            'stop' : True
            }
    else:
        if y is None:
            y = x
        diff = y-x0
        ndx = np.linalg.norm(diff)
        lr = abs(np.sum(diff*ddfx))/nddfx**2
        x0 = x
        dfx0 = dfx
        dx = -lr*dfx
        y = x+dx
        if p is None:
            step = ndx
            x = y
        else:
            x = p(y)
            step = np.linalg.norm(x-x_initial)
        out = {
            'lr' : lr,
            'ndx' : ndx,
            'step' : step,
            'y' : y,
            'x0' : x0,
            'dfx0' : dfx0,
            'stop' : False
        }
    return x, out

def mm(x,dfx,df0x=None,x0=0,df0x0=0,p=None,y=0,ndx=None,lr=10,theta=np.Inf,
       alpha=1.0,**kwargs):
    """\
    Malitsky and Mishchenko (2019) adaptive GD scheme (algorithm 4) plus
    option for projections.

    x : current position
    dfx : gradient at current position, using current approximation
    df0x : gradient at current position (or pre-projection), using old 
    approximation
    """
    if ndx is None:
        if p is None:
            ndx = np.linalg.norm(x-x0)
        else:
            ndx = np.linalg.norm(y-x0)
    if ndx == 0:
        out = {
            'stop' : True
            }
    else:
        if df0x is None:
            df0x = dfx
        nddfx = np.linalg.norm(df0x-df0x0)

        L = nddfx/ndx
        lr0 = lr
        #lr = min(math.sqrt(1+theta)*lr,1/(alpha*L))
        lr = max(min(math.sqrt(1+theta)*lr,1/(alpha*L)),3*lr/4)
        theta = lr/lr0
        dx = -lr*dfx
        ndx = np.linalg.norm(dx)
        y = x + dx
        if p is None:
            step = ndx
            x = y
        else:
            x0 = x
            x = p(y)
            step = np.linalg.norm(x-x0)
        out = {
            'ndx' : ndx,
            'df0x0' : dfx,
            'lr' : lr,
            'step' : step,
            'theta' : theta,
            'y' : y,
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

schemes = {
    'fixed' : fixed,
    'bb' : bb,
    'mm' : mm,
    'adam' : adam
    }

### ALGORITHMS ###
    
def single(x,F,Xi=None,p=None,scheme='mm',min_cost=None,
           min_grad=None, min_step=None,max_iter=100,max_step=1e10,
           lr=1,verbose=0,indent='',plot=False,**kwargs):
    """\
    Gradient descent algorithm, with different options for update rule and 
    stochastic and/or projected variaties.

    Parameters:

    x : array
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

    scheme : string
    Algorithm stepping scheme.
    """
    if Xi is None:
        stochastic = False
    else:
        stochastic = True
    if p is None:
        constraint = False
    else:
        constraint = True
    assert scheme in schemes
    algorithm = schemes[scheme]
    
    if verbose > 0:
        print(indent+'gd.single(): ')
        print(indent+'  computation parameters:')
        print(indent+f'    stochastic : {stochastic}')
        print(indent+f'    constraint : {constraint}')
        print(indent+f'    scheme : {scheme}')
        print(indent+f'    initial lr : {lr:0.2e}')
        if min_cost is not None:
            print(indent+f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(indent+f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(indent+f'    min_step : {min_step:0.2e}')
        print(indent+f'    max_iter : {max_iter}')
        print(indent+f'    max_step : {max_step:0.2e}')
        
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

    success = True
    conclusion = 'maximum number of iterations reached'
    
    t0 = time.time()

    normalization = math.sqrt(np.size(x)) ###############

    #initialization by running one iteration of GD w/ initial lr
    x0 = x.copy()
    it0 = 1
    for i in range(it0):
        if stochastic is False:
            dfx, fx = F(x)
        else:
            xi = Xi()
            dfx, fx = F(x,**xi)
        x, kwargs = fixed(x,dfx,lr=lr,p=p)
        costs[i] = fx
        grads[i] = np.linalg.norm(dfx)/normalization #######
        lrs[i] = kwargs['lr']
        steps[i] = kwargs['ndx']/normalization #rms of step size
    if constraint is True:
        y = kwargs['y']

    if verbose > 1:
        print(indent+'  progress:')
        print(indent+'    iter:      cost:     grad:     lr:       step:')
        
    for i in range(it0,max_iter):

        if stochastic is False:
            if constraint is True:
                if scheme in ['bb','mm']:
                    dfy, fy = F(y)
                    kwargs['df0x'] = dfy
            dfx, fx = F(x)
        else:
            if constraint is False:
                if scheme in ['bb','mm']:
                    df0x, f0x = F(x,**xi)
                    kwargs['df0x'] = df0x
            else:
                if scheme in ['bb','mm']:
                    fy, dfy = F(y,**xi)
                    kwargs['df0x'] = dfy
            xi = Xi()
            dfx, fx = F(x,**xi)
        costs[i] = fx
        grads[i] = np.linalg.norm(dfx)/normalization #rms of gradient
        
        if fx < min_cost:
            conclusion = 'minimum cost reached'
            lrs[i] = None
            steps[i] = None
            break
        if grads[i] < min_grad:
            conclusion = 'minimum gradient size reached'
            lrs[i] = None
            steps[i] = None
            break
        
        x, kwargs = algorithm(x,dfx,p=p,**kwargs)
        
        if constraint is True:
            y = kwargs['y']
        if kwargs['stop'] == True:
            conclusion = 'update rule'
            break
        lrs[i] = kwargs['lr']
        lr = lrs[i]
        steps[i] = kwargs['ndx']/normalization #rms of step size
        if steps[i] < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif steps[i] > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 1:
            print(indent+f'    {i+1:>4}/{max_iter:>4}  {costs[i]:0.2e}'+
                  f'  {grads[i]:0.2e}  {lrs[i]:0.2e}'+
                  f'  {steps[i]:0.2e}',flush=True, end="\r")
    
    tf = time.time()

    costs = costs[0:i+1]
    grads = grads[0:i+1]
    lrs = lrs[0:i+1]
    steps = steps[0:i+1]

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
        'iterations' : i+1,
        'success' : success,
        'conclusion' : conclusion,
        'time' : tf-t0,
        'lr' : lr
        }
        
    if verbose > 0:
        if verbose > 1:
            print()
        print(indent+'  results:')
        print(indent+f'    conclusion : {conclusion}')
        print(indent+f'    total iterations : {i+1}')
        print(indent+f'    final cost : {costs[-1]:0.2e}')
        print(indent+f'    final gradient size : {grads[-1]:0.2e}')        
        print(indent+f'    final learning rate : {lrs[-1]:0.2e}')
        print(indent+f'    final step size : {steps[-1]:0.2e}')
        print(indent+f'    time : {tf-t0:0.2e} [sec]')
        
    return x, outputs

def multiple(X,F,Xi=None,p=None,scheme='mm',min_cost=None,
             min_grad=None, min_step=None,max_iter=100,max_step=1e10,
             lr=1,verbose=0,indent='',plot=False,**kwargs):
    """\
    Gradient descent algorithms.
    """
    assert isinstance(X,list); K = len(X)

    if Xi is None:
        stochastic = False
    else:
        stochastic = True
        
    if isinstance(p,list):
        assert len(p) == K
    else:
        p = [p]*K
    projected = [pk is not None for pk in p]
    if True in projected:
        constraint = True
    else:
        constraint = False
        
    if isinstance(scheme,list):
        assert len(scheme) == K
    else:
        scheme = [scheme]*K
    if isinstance(lr,list):
        assert len(lr) == K
    else:
        lr = [lr]*K

    algorithm = []
    for k in range(K):
        algorithm.append(schemes[scheme[k]])

    if verbose > 0:
        print(indent+f'gd.multiple(): ')
        print(indent+f'  computation parameters:')
        print(indent+f'    stochastic : {stochastic}')
        print(indent+f'    constraint : {constraint}')
        print(indent+f'    projected : {projected}')
        print(indent+f'    scheme : {scheme}')
        lrs = ', '.join(f'{a:0.2e}' for a in lr)
        print(indent+f'    initial lr : {lrs}')
        if min_cost is not None:
            print(indent+f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(indent+f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(indent+f'    min_step : {min_step:0.2e}')
        print(indent+f'    max_iter : {max_iter}')
        print(indent+f'    max_step : {max_step:0.2e}')
        
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

    success = True
    conclusion = 'maximum number of iterations reached'

    t0 = time.time()

    normalization = [math.sqrt(np.size(a)) for a in X] ####

    X0 = X.copy()
    it0 = 1
    for i in range(it0):
        if stochastic is False:
            dfX, fX = F(X)
        else:
            xi = Xi()
            dfX, fX = F(X,**xi)
        KWARGS = []
        if constraint is True:
            Y = []
        for k in range(K):
            X[k], temp = fixed(X[k],dfX[k],p=p[k],lr=lr[k])
            #temp['df0x0'] = dfX[k]
            KWARGS.append(temp)
            if constraint is True:
                Y.append(temp['y'])
        costs[i] = fX
        grads[i] = [np.linalg.norm(dfX[k])/normalization[k] for k in range(K)] ####
        lrs[i] = lr
        steps[i] = [KWARGS[k]['ndx']/normalization[k] for k in range(K)]
        
    if verbose > 0:
        print(indent+'  progress:')
        print(indent+'    iter:      cost:     grad:     lr:       step:')
        
    for i in range(it0,max_iter):

        if stochastic is False:
            if constraint is True:
                dfY, fY = F(Y)
            dfX, fX = F(X)
        else:
            if constraint is False:
                df0X, f0X = F(X,**xi)
            else:
                df0Y, f0Y = F(Y,**xi)
            xi = Xi()
            dfX, fX = F(X,**xi)
            
        costs[i] = fX     
        grads[i] = [np.linalg.norm(a)/b for a, b in zip(dfX,normalization)]

        if fX < min_cost:
            conclusion = 'minimum cost reached'
            lrs[i] = [None]*K
            steps[i] = [None]*K
            break
        if max(grads[i]) < min_grad:
            conclusion = 'minimum gradient size reached'
            lrs[i] = [None]*K
            steps[i] = [None]*K
            break

        for k in range(K):
            if stochastic is False:
                if projected[k] is False:
                    KWARGS[k]['df0x'] = dfX[k]
                else:
                    KWARGS[k]['df0x'] = dfY[k]
            else:
                if constraint is False:
                    KWARGS[k]['df0x'] = df0X[k]
                else:
                    KWARGS[k]['df0x'] = df0Y[k]

            X[k], KWARGS[k] = algorithm[k](X[k],dfX[k],p=p[k],**KWARGS[k])
            if constraint is True:
                Y = [KWARGS[k]['y'] for k in range(K)]
            if KWARGS[k]['stop'] == True:
                conclusion = 'update rule'
                break
            lrs[i,k] = KWARGS[k]['lr']
            lr[k] = lrs[i,k]
            steps[i,k] = KWARGS[k]['ndx']/normalization[k]
        if max(steps[i]) < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif max(steps[i]) > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 0:
            #sys.stdout.write("\033[K")
            print(indent+f'    {i+1:>4}/{max_iter:>4}  {costs[i]:0.2e}'+\
                  f'  {np.max(grads[i]):0.2e}  {np.max(lrs[i]):0.2e}'+\
                  f'  {np.max(steps[i]):0.2e}',
                  flush=True, end="\r")
    
   # if verbose > 0:
    #    sys.stdout.write("\033[K")
     #   sys.stdout.write("\033[F")
      #  sys.stdout.write("\033[K")
        
    tf = time.time()

    costs = costs[0:i+1]
    grads = grads[0:i+1]
    lrs = lrs[0:i+1]
    steps = steps[0:i+1]

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
        'iterations' : i+1,
        'success' : success,
        'conclusion' : conclusion,
        'time' : tf-t0,
        'lr' : lr
        }
        
    if verbose > 0:
        if verbose > 1:
            print()
        print(indent+'  results:')
        print(indent+f'    conclusion : {conclusion}')
        print(indent+f'    total iterations : {i+1}')
        print(indent+f'    final cost : {costs[-1]:0.2e}')
        print(indent+f'    time : {tf-t0:0.2e} [sec]')
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
