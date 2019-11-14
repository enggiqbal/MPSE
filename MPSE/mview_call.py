import matplotlib.pyplot as plt
import numpy as np
import mview

def mds(D,dim=3,verbose=0,**kwargs):
    """\
    Run basic MDS algorithm.
    
    Parameters:

    D : numpy array
    Distance/dissimilarity matrix.
    """
    mds = mview.mds.MDS(D,dim=dim,verbose=verbose)
    mds.initialize()
    if verbose > 0:
        mds.figureX(title='Initial embedding')
    mds.optimize(algorithm='agd',verbose=verbose,**kwargs)
    if verbose > 0:
        mds.figureX(title='Final embedding')
        mds.figureH()
        plt.show()
        
    points = mds.X
    cost = mds.cost
    costhistory = mds.H['cost']

    return points, cost, costhistory

def standard(D,verbose=0,**kwargs):
    """\
    Run multiview-MDS algorithm, optimizing for data positions only.
    
    Parameters:

    D : list
    List of distance/dissimilarity matrices. Include 1 or 2 or 3.

    verbose: 0 or 1
    """

    persp = mview.perspective.Persp() #set class of projection functions
    persp.fix_Q(special='standard',number=len(D)) #fix projections
    
    mv = mview.multiview.Multiview(D,persp,verbose=verbose) #set multiview alg
    mv.setup_visualization(visualization='mds') #use mds
    mv.initialize_X(method='mds',max_iters=50) #find initial positions using mds on average distances
    mv.optimize_X(save_cost=True,verbose=verbose,
                  **kwargs) 
    #optimize for X (adaptive GD is default)
    
    points = mv.X
    cost = mv.cost
    costhistory = mv.H['cost']

    Q = mv.Q
    proj = []
    for q in Q:
        proj.append(q.T @ q)



    if verbose > 0: #plot embeddings
        mv.figureX()
        mv.figureY()
        mv.figureH()
        plt.show()

    return points,proj,cost,costhistory,

def main(D,verbose=0,**kwargs):
    """\
    Run multiview-MDS algorithm, optimizing for both data positions and
    projections.

    Parameters:

    D : list of (N x N) numpy arrays
    Distance matrices for each of three projections

    Outputs :

    X : numpy array
    Positions

    P : numpy array
    Projection matrices
    """
    persp = mview.perspective.Persp()
    #print(persp)
    mv = mview.multiview.Multiview(D,persp=persp,verbose=verbose)
    mv.setup_visualization(visualization='mds')
    mv.initialize_Q(); mv.initialize_X()
    mv.optimize_all(verbose=verbose,**kwargs)

    points = mv.X
    Q = mv.Q
    projections = []
    for q in Q:
        projections.append(q.T @ q)
    cost = mv.cost
    costhistory=np.array([0])
    if 'cost' in mv.historyX:
        costhistory = mv.historyX['cost']

    if verbose > 0:
        mv.figureX(plot=True)
        mv.figureY(plot=True)
        plt.show()

    return points,projections,cost,costhistory

def example_mds(number_of_points=990,dim=3,**kwargs):
    
    path = 'datasets/dataset_tabluar/data/'
    D = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
    sub = range(number_of_points); D = (D[sub])[:,sub] #subsample
    D += 1e-5

    points, cost, costhistory = mds(D,dim,**kwargs)
    return points, cost, costhistory

# def example_standard(number_of_points=990,number_of_projs=3,**kwargs):
#     assert number_of_points <= 990 #number of points used
#     assert number_of_projs <= 3
    
#     path = 'datasets/dataset_tabluar/data/'
#     D1 = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
#     D2 = np.genfromtxt(path+'discredit3_1000_2.csv', delimiter=',')+0.001
#     D3 = np.genfromtxt(path+'discredit3_1000_3.csv', delimiter=',')+0.001
    
#     sub = range(number_of_points) #subsample data
#     D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub]
#     D = [D1,D2,D3]; D = D[0:number_of_projs]
    
#     points,proj,cost,costhistory=standard(D,**kwargs)
#     return points,proj,cost,costhistory

# def example_main(number_of_points=990,number_of_projs=3,**kwargs):
#     path = 'datasets/dataset_tabluar/data/'
#     D1 = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
#     D2 = np.genfromtxt(path+'discredit3_1000_2.csv', delimiter=',')+0.01
#     D3 = np.genfromtxt(path+'discredit3_1000_3.csv', delimiter=',')+0.01
    
#     sub = range(number_of_points) #subsample of data
#     D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub]
#     D = [D1,D2,D3]; D = D[0:number_of_projs]
    
#     points,projections,cost,costhistory=main(D,**kwargs)
#     return points,projections,cost,costhistory

def example_standard(number_of_points=100,number_of_projs=2,**kwargs):
    path = 'datasets/dataset_3D/circle_square_new/'
    D1 = np.genfromtxt(path+'dist_circle.csv', delimiter=',')
    D2 = np.genfromtxt(path+'dist_square.csv', delimiter=',')
#    D3 = np.genfromtxt(path+'discredit3_1000_3.csv', delimiter=',')+0.01
    
    sub = range(number_of_points) #subsample of data
    #D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub]
#    D = [D1,D2,D3]; D = D[0:number_of_projs]
    D = [D1,D2]; D = D[0:number_of_projs]
    points,projections,cost,costhistory=main(D,**kwargs)
    return points,projections,cost,costhistory
    
if __name__=='__main__':
    example_mds(100,verbose=1,min_step=1e-4)
    #example_standard(100,verbose=1)
    example_standard(100,2,verbose=1)
    #example_standard(990,2,min_step=1e-4)
    #example_main(100,2,verbose=1,max_iters=300,min_step=1e-5,algorithm='agd')
    #example_main(100,3,min_step=1e-4,rounds=10)
 
