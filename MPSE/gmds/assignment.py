import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def random_permutation(n):
    """\
    Return random permutation of [1,2,...,n]
    """
    w = list(range(n))
    random.shuffle(w)
    return w

def invert_permutation(w):
    """\
    Return the inverse of the given permutation

    --- arguments ---
    w = list containing permutation of [1,2,...,len(perm)]
    """
    n = len(w)
    inv = [0]*n
    for i in range(n):
        inv[w[i]] = i
    return inv

def permute_matrix(D0,w):
    """\
    Return distance matrix with rows and columns permutted by w, as in
    D = WDW^T, where W is obtained from the identity matrix by permuting rows
    in the order given by perm.
    """
    D = D0[w,:][:,w]
    return D
        
def stress(d,D0,w=None):
    n = len(d)
    if w is None:
        w = list(range(n))
        D = D0
    else:
        D = permute_matrix(D0,w)
    total = 0
    for i in range(n):
        for j in range(i+1,n):
            total += (d[i,j]-D[i,j])**2
    return total

def correlation(d,D0,w=None):
    """\
    Correlation between distance matrices d and D0. If permutation perm is 
    included, row and columsn of D are permuted by w.

    --- arguments ---
    d = distance matrix (corresponding to embedded points after projection)
    D = distance matrix (corresponding to target)
    perm = permutation of indices [1,2,...,len(d)]
    """
    n = len(d)
    if w is None:
        w = list(range(n))
        D = D0
    else:
        D = permute_matrix(D0,w)
        total = 0
    for i in range(n):
        for j in range(i+1,n):
            total += d[i,j]*D[i,j]
    return total

def swap(i1,i2,w):
    """\
    Returns permutation after swapping values on indices i1 and i2.
    """
    wi1 = w[i1]
    wi2 = w[i2]
    w_swap = w.copy()
    w_swap[i1] = wi2
    w_swap[i2] = wi1
    return w_swap

def gain(i1,i2,d,D0,w=None):
    """\
    Gain in correlation when swapping the i1 and i2 rows/columns of D. D is
    first permutted by perm if included.

    --- arguments ---
    i1, i2 = indeces of rows/columns
    d, D = distance matrices
    perm = permutation
    """
    n = len(d)
    if w is None:
        w = list(range(n))
    #total = 0
    #for i in [i for ii, i in enumerate(range(n)) if ii not in [i1,i2]]:
    #    temp = d[perm[i2],perm[i]]-d[perm[i1],perm[i]]
    #    total += temp*(D[i1,i]-D[i2,i])
    #print(total)
    w_swap = swap(i1,i2,w)
    #total = correlation(d,D,perm2)-correlation(d,D,perm)
    total = -(stress(d,D0,w_swap)-stress(d,D0,w))
    return total

def swap_up(i1,perm,d,D0,indices=None):
    """\
    Update perm by swapping i1 with another index that increases correlation
    between d and D. If a subset of indices is given, the swap can only occur
    with indices in the subset. The order in which the indices are checked for
    improvement is random.
    """
    n = len(d)
    if indices is None:
        indices = list(range(n))
    random.shuffle(indices)
    for k in range(len(indices)):
        i2 = indices[k]
        g = gain(i1,i2,d,D0,w=perm)
        if g > 0:
            perm = swap(i1,i2,perm)
            print(stress(d,D0,perm))
            break        
    return perm

def neighborhoods(d,nneighs):
    """\
    Return matrix containing indices of neighbors for each index in distance
    matrix.
    
    --- arguments ---
    d = distance matrix
    nneighs = number of points in each neighborhood
    """
    n = len(d)
    neighs = np.empty((n,nneighs),dtype=int)
    for i in range(n):
        neighs[i] = np.argsort(d[i])[1:nneighs+1]
    return neighs

def update_random_switches(d,D0,perm=None,iterations=1000,nneighs=None):
    """\
    Return random improvement of perm
    """
    n = len(d)
    if perm is None:
        perm = range(n)
    if nneighs is None:
        nneighs = n-1
    neighs = neighborhoods(d,nneighs)
    
    for i in range(iterations):
        i1 = random.randint(0,n-1)
        perm = swap_up(i1,perm,d,D0,indices=neighs[i1])

    return perm

def reassign(ds,Ds,iteratiosn=1000,nneighs=None):
    """\
    Return permutations of nodes for multiple distance/target pairs
    """
    K = len(ds)
    ps = []
    for k in range(K):
        ps += [update_random_switches(ds[k],Ds[k],perm=p0s[k],
                                     iterations=1000,nneighs=nneighs)]
    return ps

########## Tests ##########

def test0():
    """\
    Small test
    """
    n = 10
    perm0 = list(range(n))
    perm = random_permutation(n)
    
    print(perm0)
    print(perm)
    x = np.random.rand(n,2)
    D = distance_matrix(x,x)
    d = permute_matrix(D,perm)
    print('stress',stress(d,D,perm0))
    perm1 = update_random_switches(d,D,perm=perm0,nneighs=5,iterations=10000)
    
    print('original:',perm0)
    print('target:',perm)
    print('result:',perm1)
    
    

    
