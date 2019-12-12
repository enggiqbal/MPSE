import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix

from . import mds, special

### MDSP ###

# Note: this section contains examples of convergence of gradient descent for
# MDSp stress for fixed positions X and varying map P

def MDSp_example0():
    """\
    Example of MDSp algorithm for exact random points in plane

    The positions X in R3 are random and fixed. 
    The map P is projection onto the yz plane.
    The initial map P0 is choosen randomly.
    """
    n = 10 #number of points
    X = np.random.rand(n,3) #positions
    P = np.diag([0,1,1]) #map
    PX = X @ P.T #new positions
    D = distance_matrix(PX,PX) #distances of new positions

    P0 = np.random.rand(3,3) #initial map
    P1 = mds.MDSp_Pdescent(X,D,P0,feedback=True,rate=.05,max_iters=10000)
    
    print('\nExample of MDSp optimization for data in cube')
    print('n = 10, rate=0.05, max_iters=10000')
    print('\nTrue map is P =')
    print(P)
    print('Initial map is P0 = ')
    print(P0)
    print('Final map is P1 =')
    print(P1)
    print('P1 @ P1.T =')
    print(P1.T @ P1)

def MDSp_rate(n=10,min_rate=-6,max_rate=-2):
    """\
    Stress for different learning rates
    """
    rates = 2.0**np.arange(min_rate,max_rate)

    X = np.random.rand(n,3) #positions
    P = np.diag([0,1,1]) #map
    PX = X @ P.T #new positions
    D = distance_matrix(PX,PX) #distances of new positions
    P0 = np.random.rand(3,3) #initial map

    plt.figure()
    for i in range(len(rates)):
        P_trajectory = mds.MDSp_Pdescent(X,D,P0,rate=rates[i],trajectory=True)
        
        m = len(P_trajectory)
        stress = np.empty(m)
        for j in range(m):
            stress[j] = mds.MDSp_stress(X,P_trajectory[j],D)

        plt.plot(stress,label=f'2^{i+min_rate}')

    plt.title(f'MDSp stress for different learning rates, n = {n}')
    plt.xlabel('iteration number')
    plt.yscale('log')
    plt.ylim(top=1e5)
    plt.legend()
    plt.show()

def MDSp_initial(n=10, rate=2**-4, runs=10):
    """\
    Convergence of MDSp algorithm for uniformly sampled points for different
    initial conditions.

    n : number of points
    runs : number of runs
    """
    print('\n### Begin MDSp_initial ###')
    X = np.random.rand(n,3) #positions
    P = np.diag([0,1,1]) #map
    PX = X @ P.T #new positions
    D = distance_matrix(PX,PX) #distances of new positions

    plt.figure()
    for i in range(runs):
        print(f'run {i+1} out of {runs}')
        P0 = np.random.rand(3,3)
        P_trajectory = mds.MDSp_Pdescent(X,D,P0,trajectory=True,rate=rate)

        m = len(P_trajectory)
        stress = np.empty(m)
        for j in range(m):
            stress[j] = mds.MDSp_stress(X,P_trajectory[j],D)
        plt.plot(stress)
    
    plt.title(f'MDSp stress for different initial conditions, n = {n}.')
    plt.xlabel('iteration number')
    plt.ylabel('stress')
    plt.yscale('log')
    plt.show()
    print('### End MDS_convergence_initial ###')

### multiMDSp X descent ###

def mMDSp_Xdescent_example0():
    """\
    Example of multiMDSp algorithm for exact random points in cube

    The positions X in R3 are random. 
    The maps P1, P2, P3 are the standard projections.
    """
    print('\nExample of multiMDSp X optimization for data in cube')
    print('n=10, rate=0.01')
    n = 10 #number of points

    X = np.random.rand(n,3) #true positions
    PP = np.zeros((3,3,3)) #list of maps
    DD = np.zeros((3,n,n)) #list of map distances
    for k in range(3): 
        PP[k] = np.diag([1,1,1]); PP[k,k,k] = 0 #maps are standard projections
        temp = X @ PP[k].T; DD[k] = distance_matrix(temp,temp) #map distances

    X0 = np.random.rand(n,3) #initial positions
    X1 = mds.mMDSp_Xdescent(PP,DD,X0,feedback=True,rate=0.01) #solution

def mMDSp_Xrate(n=10,min_rate=-8,max_rate=-2):
    """\
    Stress for different learning rates
    """
    print('\nRates of mMDSx for data in cube')
    rates = 2.0**np.arange(min_rate,max_rate)

    X = np.random.rand(n,3) #true positions
    PP = np.zeros((3,3,3)) #list of maps
    DD = np.zeros((3,n,n)) #list of map distances
    for k in range(3): 
        PP[k] = np.diag([1,1,1]); PP[k,k,k] = 0 #maps are standard projections
        temp = X @ PP[k].T; DD[k] = distance_matrix(temp,temp) #map distances
    X0 = np.random.rand(n,3) #initial positions
    
    X1 = mds.mMDSp_Xdescent(PP,DD,X0,feedback=True,rate=0.01) #solution
    plt.figure()
    for i in range(len(rates)):
        X_trajectory = mds.mMDSp_Xdescent(PP,DD,X0,rate=rates[i],
                                            trajectory=True)
        m = len(X_trajectory)
        stress = np.empty(m)
        for j in range(m):
            stress[j] = mds.mMDSp_stress(X_trajectory[j],PP,DD)
        plt.plot(stress,label=f'2^{i+min_rate}')

    plt.title(f'multi-MDS stress optimization for X, n = {n}')
    plt.xlabel('iteration number')
    plt.yscale('log')
    plt.ylim(top=1e5)
    plt.legend()
    plt.show()

### mMDSp XP optimization ###

def mMDSp_XPexample():
    """\
    Example of multiview MDS optimization for X and P for exact random points 
    in cube

    The positions X in R3 are random. 
    The maps P1, P2, P3 are the standard projections.
    """
    print('\nExample of mMDSp XP optimization for data in cube')
    print('n=10, rate=0.01')
    n = 10 #number of points

    X = np.random.rand(n,3) #true positions
    PP = np.zeros((3,3,3)) #list of maps
    DD = np.zeros((3,n,n)) #list of map distances
    for k in range(3): 
        PP[k] = np.diag([1,1,1]); PP[k,k,k] = 0 #maps are standard projections
        temp = X @ PP[k].T; DD[k] = distance_matrix(temp,temp) #map distances

    X0 = np.random.rand(n,3) #initial positions
    PP0 = []
    for k in range(3):
        PP0 += [np.random.rand(3,3)] #initial maps
        
    rates = [0.005]+3*[0.01]
    X1 = mds.mMDSp_XPdescent(X0,PP0,DD,feedback=True,rates=rates) #solution
    
### special ###

def mMDSq_Xdescent_Qmultiple_example0():
    """\
    Example of multiview-MDSq optimization for X using various Q_list

    The positions X in R3 are random. 
    The true projections are the standard projections
    """
    print('\nExample of multiview-MDSq X optimization for various Q_list')
    print('n=10, rate=0.01')
    n = 10 #number of points

    X = random_ssphere(n) #true positions
    Q_list = special.standard_projection_matrices()
    D_list = special.distance_matrices(X,Q_list)

    X0 = random_ssphere(n) #initial positions

    X, Q_list = special.mMDSq_Xdescent_Qmultiple(X0,D_list,runs=500,
                                                 feedback=True,rate=0.01,
                                                 max_iters=1000)
    print(Q_list)

def mMDSq_Xdescent_noisyQ0(n=10,max_iters=500,rate=0.01):
    """\
    """
    print('\nMultiview-MDSq X optimization for points in solid sphere and standard projections. The initial points are the true points X and the projections used are the true projections plus entry-wise normal noise projected into orthogonal matrices.')

    X = random_ssphere(n)  #true positions
    Q_list = special.standard_projection_matrices()
    D_list = special.distance_matrices(X,Q_list)
    print(mds.mMDSq_stress(X,Q_list,D_list))
    print('hi')
    noise = [0,0.05,0.1,0.15,0.2,0.3,0.5,1.0]
    stress0 = []; stress1 = []; dist = []
    for i in range(len(noise)):
        Q0_list = special.noisy_projection_matrices(Q_list,sigma=noise[i])
        print(Q0_list)
        X1 = mds.mMDSq_Xdescent(X,Q0_list,D_list,feedback=True,
                                 rate=rate,max_iters=max_iters) 
        stress0 += [mds.mMDSq_stress(X,Q0_list,D_list)]
        stress1 += [mds.mMDSq_stress(X1,Q0_list,D_list)]
        dist += [special.distance_between_projections(Q_list,Q0_list)]

    plt.figure()
    plt.plot(noise,stress0,label='true X')
    plt.plot(noise,stress1,label='optimal X')
    plt.plot(noise,dist,label='distance between proj')
    plt.xlabel('noise level')
    plt.ylabel('stress')
    plt.legend()
    plt.show()
    return
