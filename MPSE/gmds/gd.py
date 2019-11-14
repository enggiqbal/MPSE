import numpy as np
import matplotlib.pyplot as plt
import copy

### Gradient descent ###

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
    f = lambda x: x[0]**2
    df = lambda x: 2*x
    results = gradient_descent(x0,df,f=f,rate=0.01,step_history=True,
                               cost_history=True,feedback=True,
                               plot_history=True)
    x = results['output'][0]
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
    results = gradient_descent(x0,df,projection=p,f=f,rate=0.01,
                               step_history=True,cost_history=True,
                               feedback=True,plot_history=True)
    x = results['output']
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
    
############################### Old ###############################

def coord_gradient_descent(df_list,x0_list,**kwargs):
    """\
    Coordinate gradient descent

    df_list : list containing gradient functions for each component
    x0_list : list containing starting positions for each component
    """
    params = {
        'rate' : 1.0, #learning rate
        'precision' : 1e-6, #min step size stopping criteria
        'max_iters' : 1000, #max number of iterations
        'max_step_size' : 1e5, #max step size stopping criteria

        'coord_rates' : None, #learning rates for coord gradient descent
        'coord_max_iters' : 10, #max number of iterations for each coordinate
    }
    params.update(kwargs)
    if params['coord_rates'] is None:
        params['coord_rates'] = len(dff)*[params['rate']]
    if isinstance(params['coord_max_iters'],int):
        params['cord_max_iters'] = len(dff)*[params['coord_max_iters']]

    M = len(df_list) #number of coordinates
    iters = 0 #number of iterations
    step_size = 1.0 #initial step size
    xx = x0_list.copy()
    
    while (params['precision'] < step_size < params['max_step_size'] and
           iters < params['max_iters']):
        step_size = 0
        for m in range(M):
            xx0 = xx.copy()
            def dfm(xxm):
                args = xx0.copy()
                args[m] = xxm
                return dff[m](args)
            xx[m] = gradient_descent(dfm,xx0[m],rate=params['rates'][m],
                                     max_iters=params['coord_max_iters'][m])
            step_size += np.linalg.norm(xx[m]-xx0[m]) 
        iters += 1

    return xx

def coord_projected_gradient_descent(df_list,p_list,x0_list,**kwargs):
    """\
    Projected coordinate gradient descent
    
    df_list : list containing gradient functions
    p_list : list containing projection functions
    x0_list : list containing initial configurations
    """
    params = {
        'rate' : 1.0, #learning rate
        'precision' : 1e-6, #min step size stopping criteria
        'max_iters' : 100, #max number of iterations
        'max_step_size' : 1e5, #max step size stopping criteria

        'coord_rates' : None, #learning rates for coord gradient descent
        'coord_max_iters' : 100, #max number of iterations for each coordinate
    }
    params.update(kwargs)
    if params['coord_rates'] is None:
        params['coord_rates'] = len(df_list)*[params['rate']]
    if isinstance(params['coord_max_iters'],int):
        params['coord_max_iters'] = len(df_list)*[params['coord_max_iters']]
    M = len(df_list)
    iters = 0
    step_size = 1.0
    xx = x0_list.copy()

    while (params['precision'] < step_size < params['max_step_size'] and
           iters < params['max_iters']):

        step_size = 0
        for m in range(M):
            xx0 = xx.copy()

            def dfm(xxm):
                args = xx0.copy()
                args[m] = xxm
                return df_list[m](args)
            
            xx[m] = projected_gradient_descent(dfm,p_list[m],xx[m],
                                               rate=params['coord_rates'][m],
                                               max_iters=params['coord_max_iters'][m])
            step_size += np.linalg.norm(xx[m]-xx0[m])
        iters += 1

    return xx
