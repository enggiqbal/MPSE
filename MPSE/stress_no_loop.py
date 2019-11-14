import torch
import networkx as nx
import time
import numpy as np
# cuda = torch.device('cpu')
cuda = torch.device('cuda')
#cuda = torch.device('cuda:0')
#cuda = torch.device('cuda:1')

#https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def dict2tensor(d):
    res = None
    for k,v in d.items():
        for k2,v2 in v.items():
            if res is None:
                res = torch.zeros(len(d.keys()), len(v.keys()), device=cuda)
            res[k,k2] = v2
    return res


def stress(x, d):
#     s = torch.zeros(1, device=cuda)
#     torch.cuda.synchronize()
#     for i in range(len(x)):
#         for j in range(i):
#             de = torch.sqrt(
#                     torch.sum(
#                         (x[i,0:2]-x[j,0:2])**2
#                 ))
#             s += (de - d[i][j])**2
#     torch.cuda.synchronize()
    pdist = pairwise_distances(x)
    s = ((pdist - d)**2).mean()
    return s


def stress_minimization(X, D, lr=.01, prec=.001, max_iter=1000):
    step = 1
    i = 0
    while i<max_iter:
        s = stress(X, D)
#         print(s.item())
        s.backward()
        with torch.no_grad():
            X = X - lr*X.grad
        X.requires_grad = True
        i += 1
#     torch.cuda.synchronize()
    print('i:', i)
    print('X:', X)

def test_stress_min():
    #n = 5
    #n = 10
    #X = torch.rand(n, 2, requires_grad = True)
    #G = nx.path_graph(n)
    #G = nx.cycle_graph(n)
    r = 2
    h = 5
    #h = 8
#     torch.cuda.init()
    X = torch.rand(r**(h+1)-1, 2, requires_grad = True, device=cuda)
#     torch.cuda.synchronize()
    G = nx.balanced_tree(r, h)
    D = dict(nx.all_pairs_shortest_path_length(G))
    D = dict2tensor(D)
#     print('X:', X)
#     print('D:', D)
    stress_minimization(X, D, max_iter=1000)

start_time = time.time()
test_stress_min()
print('Running time:', (time.time() - start_time))


