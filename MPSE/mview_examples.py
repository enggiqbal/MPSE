import mview

import numpy as np

### MDS examples ###

def mds_disk(N=100,dim=2):
    print('\n***mds.example_disk()***')
    
    Y = mview.misc.disk(N,dim); labels = mview.misc.labels(Y)   
    D = mview.distances.compute(Y)
    mview.MDS(D,dim,verbose=2,title='random data on disk',plot=True,
              labels=labels)

def mds_credit(number_of_points=990,dim=3,**kwargs):

    path = 'datasets/dataset_tabluar/data/'
    D = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
    sub = range(number_of_points); D = (D[sub])[:,sub] #subsample
    points, cost, costhistory = mview.MDS(D,dim,verbose=2,plot=True,**kwargs)
    
### MULTIVIEW0 examples ###
    
def multiview0_credit(number_of_points=100,number_of_projs=2,**kwargs):
    path = 'datasets/dataset_tabluar/data/'
    D1 = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
    D2 = np.genfromtxt(path+'discredit3_1000_2.csv', delimiter=',')
    D3 = np.genfromtxt(path+'discredit3_1000_3.csv', delimiter=',')
    sub = range(number_of_points) #subsample of data
    D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub]
    D = [D1,D2,D3]; D = D[0:number_of_projs]
    D = [D1,D2]; D = D[0:number_of_projs]
    points,cost,costhistory=mview.MULTIVIEW0(D,verbose=2,Q='standard',
                                             plot=True,**kwargs)

### MULTIVIEW examples ###

def multiview_credit(number_of_points=100,number_of_projs=2,**kwargs):
    path = 'datasets/dataset_tabluar/data/'
    D1 = np.genfromtxt(path+'discredit3_1000_1.csv', delimiter=',')
    D2 = np.genfromtxt(path+'discredit3_1000_2.csv', delimiter=',')
    D3 = np.genfromtxt(path+'discredit3_1000_3.csv', delimiter=',')
    sub = range(number_of_points) #subsample of data
    D1 = (D1[sub])[:,sub]; D2 = (D2[sub])[:,sub]; D3 = (D3[sub])[:,sub]
    D = [D1,D2,D3]; D = D[0:number_of_projs]
    D = [D1,D2]; D = D[0:number_of_projs]
    points,projections,cost,costhistory=mview.MULTIVIEW(D,verbose=2,plot=True,
                                                        **kwargs)

if __name__=='__main__':
    #MDS:
    #mds_disk()
    mds_credit(number_of_points=100)
    multiview0_credit(number_of_points=100,number_of_projs=2)
    multiview_credit(number_of_points=100,number_of_projs=2)
