import scipy
import numpy as np

import torch

from scipy._lib._util import check_random_state
from scipy.sparse import issparse
from scipy import linalg
from scipy import spatial


from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

MACHINE_EPSILON = np.finfo(np.double).eps


n_components = 3
perplexity = 30

def projection(data, data1, data2, data3):
    P1 = np.random.rand(4, 1)
    P2 = np.random.rand(4, 1)
    P3 = np.random.rand(4, 1)
    # dim_1_D = spatial.distance_matrix(dim_1, dim_1)
    # dim_2_D = spatial.distance_matrix(dim_2, dim_2)
    # dim_3_D = spatial.distance_matrix(dim_3, dim_3)
    dim_1_D = spatial.distance_matrix(data1, data1)
    dim_2_D = spatial.distance_matrix(data2, data2)
    dim_3_D = spatial.distance_matrix(data3, data3)

    return dim_1_D, dim_2_D, dim_3_D


def prob(distances, perplexity):

    # distances_data= scipy.spatial.distance.squareform(distances)
    distances_data=distances.astype(np.float32, copy=False)
    n_samples = len(distances_data)

    lower_bound=0.01
    upper_bound=100
    iterations=10
    sigma=np.empty(n_samples)

    for i in range(n_samples):
        sigma_i=(lower_bound*upper_bound)**(1/2)
        for j in range(iterations):
            D_i=np.delete(distances_data[i],i)
            P_i=np.exp(-D_i**2/(2*sigma_i**2))
            P_i=P_i/np.sum(P_i)
            HP_i= -np.dot(P_i, np.log2(P_i +MACHINE_EPSILON))
            perplexity_i=2**HP_i

            if perplexity_i>perplexity:
                upper_bound=sigma_i
            else:
                lower_bound=sigma_i

        sigma[i]=(lower_bound*upper_bound)**(1/2)

    conditional_prob=np.exp(-distances**2)/2*sigma**2
    np.fill_diagonal(conditional_prob,0)
    conditional_prob = conditional_prob/np.sum(conditional_prob, axis=1)

    joint_P=conditional_prob+conditional_prob.T
    sum_P = np.maximum(np.sum(joint_P), MACHINE_EPSILON)
    joint_P = np.maximum(scipy.spatial.distance.squareform(joint_P) / sum_P, MACHINE_EPSILON)
    return joint_P




# def KL_divergence(P, embedding):
#     # calculation of Q
#     dist=scipy.spatial.distance.pdist(embedding,"sqeuclidean")
#     dist += 1.
#     dist **= -1
#     Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
#     kl_div=2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
#
#     return kl_div

def KL_divergence_n(params, P,P1, degrees_of_freedom, n_samples, n_components):
    # calculation of Q
    X_embedded = params.reshape(n_samples, n_components)
    X_embedded_1=np.zeros(shape=(n_samples,2))
    for i in range(len(X_embedded)):
        X_embedded_1[i]=np.matmul(P1,X_embedded[i])
    # print(X_embedded_1.shape)
    dist = scipy.spatial.distance.pdist(X_embedded_1, "sqeuclidean")
    # print("X_embedded.shape:")
    # print(X_embedded.shape)
    # dist=P1*P1*dist
    #dist /= degrees_of_freedom
    dist += 1.
    #dist **= (degrees_of_freedom + 1.0) / -2.0
    dist **= -1 ###
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = scipy.spatial.distance.squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    # print("kl_div,:")
    # print(grad)


    return kl_divergence, grad

def KL_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    # calculation of Q
    X_embedded = params.reshape(n_samples, n_components)

    dist = scipy.spatial.distance.pdist(X_embedded, "sqeuclidean")
    # print("X_embedded.shape:")
    # print(X_embedded.shape)
    # dist=P1*P1*dist
    #dist /= degrees_of_freedom
    dist += 1.
    #dist **= (degrees_of_freedom + 1.0) / -2.0
    dist **= -1 ###
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = scipy.spatial.distance.squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    # print("kl_div,:")
    # print(grad.shape)


    return kl_divergence, grad

def gradient_KL_divergence(P, embedding, proj_1):
    dist = scipy.spatial.distance.pdist(embedding, "sqeuclidean")
    dist=proj_1*proj_1*dist
    dist += 1.
    dist **= -1
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    kl_div = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    grad = np.ndarray(embedding.shape)
    PQd = scipy.spatial.distance.squareform((P - Q) * dist)

    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),embedding[i] - embedding)

    grad *= 4
    return kl_div, grad

def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7):

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it

        for i in range(it, n_iter):

            error, grad = obj_func(p, *args)
            print(grad)
            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update


            # print("[t-SNE] Iteration %d: error = %.7f,"
            #       " gradient norm = %.7f"
            #       % (i + 1, error, grad_norm))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break

            if grad_norm <= min_grad_norm:
                break
        return p


def fit(X):
    n_samples = X.shape[0]

    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)
    print("distance.shape")
    print(distances.shape)

    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    # P=prob(distances,perplexity)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

def fit_n(X, d_1, d_2, d_3):
    n_samples = X.shape[0]

    # Compute euclidean distance
    # distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    # P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    P_1 = prob(d_1,perplexity)
    P_2 = prob(d_2, perplexity)
    P_3 = prob(d_3, perplexity)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne_n(P_1,P_2, P_3, degrees_of_freedom, n_samples, X_embedded=X_embedded)


def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
        params = X_embedded.ravel()

        obj_func = KL_divergence

        params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])

        X_embedded = params.reshape(n_samples, n_components)

        return X_embedded


def _tsne_n(P_1, P_2, P_3, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()

    obj_func = KL_divergence_n

    # P1 = np.random.rand(2, 3)
    # P2 = np.random.rand(2, 3)
    # P3 = np.random.rand(2, 3)

    P1 = np.array([[1, 0, 0], [0, 1, 0]])
    P2 = np.array([[0, 1, 0], [0, 0, 1]])
    P3 = np.array([[1, 0, 0], [0, 0, 1]])
    params_1 = _gradient_descent(obj_func, params, [P_1, P1, degrees_of_freedom, n_samples, n_components])
    params_2 = _gradient_descent(obj_func, params, [P_2, P2, degrees_of_freedom, n_samples, n_components])
    params_2 = _gradient_descent(obj_func, params, [P_3, P3, degrees_of_freedom, n_samples, n_components])

    X_embedded = params.reshape(n_samples, n_components)

    return X_embedded
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


def fit_new(X, d1, d2, d3):
    n_samples = d1.shape[0]
    D=spatial.distance_matrix(X,X)
    condensed_D=spatial.distance.pdist(X)
    # P1 = np.array([[1, 0, 0], [0, 1, 0]])
    # P2 = np.array([[0, 1, 0], [0, 0, 1]])
    # P3 = np.array([[1, 0, 0], [0, 0, 1]])
    condensed_D1=spatial.distance.pdist(d1)
    condensed_D2 = spatial.distance.pdist(d2)
    condensed_D3 = spatial.distance.pdist(d3)
    P_1 = joint_probabilities(condensed_D1,n_samples,perplexity)
    P_2 = joint_probabilities(condensed_D2, n_samples, perplexity)
    P_3 = joint_probabilities(condensed_D3, n_samples, perplexity)
    # Compute euclidean distance
    # distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    # P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    # P = joint_probabilities(distances, n_samples, perplexity)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
    X_embedded_2 = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components-1).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)



    return _tsne_new(P_1,P_2,P_3, degrees_of_freedom, n_samples, X_embedded=X_embedded,X_embedded_2=X_embedded_2)

def joint_probabilities(distances, n_samples, perplexity):
    distances = scipy.spatial.distance.squareform(distances)
    lower_bound = 1e-2
    upper_bound = 1e2
    iters = 10  # parameters for binary search
    sigma = np.empty(n_samples)  # bandwith array
    for i in range(n_samples):
        # initialize bandwith parameter for sample i:
        sigma_i = (lower_bound * upper_bound) ** (1 / 2)
        for iter in range(iters):
            # distances to sample i, not including self:
            D_i = np.delete(distances[i], i)
            # compute array with conditional probabilities w.r.t. sample i:
            P_i = np.exp(-D_i ** 2 / (2 * sigma_i ** 2))
            P_i /= np.sum(P_i)  ####
            # compute perplexity w.r.t sample i:
            HP_i = -np.dot(P_i, np.log2(P_i + MACHINE_EPSILON))
            PerpP_i = 2 ** (HP_i)
            # update bandwith parameter for sample i:
            if PerpP_i > perplexity:
                upper_bound = sigma_i
            else:
                lower_bound = sigma_i
        # final bandwith parameter for sample i:
        sigma[i] = (lower_bound * upper_bound) ** (1 / 2)
    conditional_P = np.exp(-distances ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(conditional_P, 0)
    conditional_P /= np.sum(conditional_P, axis=1)

    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(scipy.spatial.distance.squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def _tsne_new(P_1,P_2, P_3, degrees_of_freedom, n_samples, X_embedded,X_embedded_2):
    params = X_embedded.ravel()
    params_2=X_embedded_2.ravel()

    P1 = np.array([[1, 0, 0], [0, 1, 0]])
    P2 = np.array([[0, 1, 0], [0, 0, 1]])
    P3 = np.array([[1, 0, 0], [0, 0, 1]])

    obj_func = KL_divergence_new
    params = _gradient_descent_new(obj_func, params,params_2, [P_1,P_2,P_3, degrees_of_freedom, n_samples, n_components])

    X_embedded = params.reshape(n_samples, 3)

    return X_embedded

def _gradient_descent_new(obj_func, p0,p1, args, it=0, n_iter=500,
                          n_iter_check=1, n_iter_without_progress=100,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7):

        p = p0.copy().ravel()
        p2=p1.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it

        for i in range(it, n_iter):

            error, grad = obj_func(p, *args)
            print(grad)
            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            # print("[t-SNE] Iteration %d: error = %.7f,"
            #       " gradient norm = %.7f"
            #       % (i + 1, error, grad_norm))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break

            if grad_norm <= min_grad_norm:
                break
        return p

def KL_divergence_new(params, P_1,P_2,P_3, degrees_of_freedom, n_samples, n_components):

    P1 = np.array([[1, 0, 0], [0, 1, 0]])
    P2 = np.array([[0, 1, 0], [0, 0, 1]])
    P3 = np.array([[1, 0, 0], [0, 0, 1]])

    # calculation of Q
    X_embedded_1 = params.reshape(n_samples, n_components)
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, 2).astype(np.float32)
    print(X_embedded_1[0].shape)
    print(P1.shape)
    for i in range(n_samples):
        X_embedded[i]=np.matmul(P1,X_embedded_1[i])

    dist = scipy.spatial.distance.pdist(X_embedded, "sqeuclidean")
    # print("X_embedded.shape:")
    # print(X_embedded.shape)
    # dist=P1*P1*dist
    #dist /= degrees_of_freedom
    dist += 1.
    #dist **= (degrees_of_freedom + 1.0) / -2.0
    dist **= -1 ###
    Q_1 = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # X_embedded = params.reshape(n_samples, n_components)
    for i in range(n_samples):
        X_embedded[i] = np.matmul(P2, X_embedded_1[i])

    dist = scipy.spatial.distance.pdist(X_embedded, "sqeuclidean")
    dist += 1.
    # dist **= (degrees_of_freedom + 1.0) / -2.0
    dist **= -1  ###
    Q_2 = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # X_embedded = params.reshape(n_samples, n_components)
    for i in range(n_samples):
        X_embedded[i] = np.matmul(P3, X_embedded_1[i])

    dist = scipy.spatial.distance.pdist(X_embedded, "sqeuclidean")
    dist += 1.
    # dist **= (degrees_of_freedom + 1.0) / -2.0
    dist **= -1  ###
    Q_3 = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence_1 = 2.0 * np.dot(P_1, np.log(np.maximum(P_1, MACHINE_EPSILON) / Q_1))
    kl_divergence_2 = 2.0 * np.dot(P_2, np.log(np.maximum(P_2, MACHINE_EPSILON) / Q_2))
    kl_divergence_3 = 2.0 * np.dot(P_3, np.log(np.maximum(P_3, MACHINE_EPSILON) / Q_3))
    kl_divergence=kl_divergence_1+kl_divergence_2+kl_divergence_3

    # Gradient: dC/dY
    grad_1 = np.ndarray((n_samples, 3), dtype=params.dtype)
    PQd = scipy.spatial.distance.squareform((P_1 - Q_1) * dist)
    for i in range(n_samples):
        grad_1[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded_1[i] - X_embedded_1)
    grad_1 = grad_1.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad_1 *= c
    # print("kl_div,:")
    # print(grad.shape)
    grad_2 = np.ndarray((n_samples, 3), dtype=params.dtype)
    PQd = scipy.spatial.distance.squareform((P_2 - Q_2) * dist)
    for i in range(n_samples):
        grad_2[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded_1[i] - X_embedded_1)
    grad_2 = grad_2.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad_2 *= c

    grad_3 = np.ndarray((n_samples, 3), dtype=params.dtype)
    PQd = scipy.spatial.distance.squareform((P_3 - Q_3) * dist)
    for i in range(n_samples):
        grad_3[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded_1[i] - X_embedded_1)
    grad_3 = grad_3.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad_3 *= c
    grad=grad_1+grad_2+grad_3


    return kl_divergence, grad

# class TSNE(object):
#
#     def __init__(self, data, dim=2, perplexity=30, sample_colors=None, verbose=0, indent='',title='', **kwargs):
#         #self.n_components = n_components
#         self.perplexity = perplexity
#         self.data=data
#         self.verbose = verbose
#
#     def initialization(self,x_true):
#         self.data=x_true
#         embedding = self._fit(x_true)
#         self.embedding_ = embedding
#         return self.embedding_
#
#     def _fit(self, X, skip_num_points=0):
#         """Private function to fit the model using X as training data."""
#
#         if self.method not in ['barnes_hut', 'exact']:
#             raise ValueError("'method' must be 'barnes_hut' or 'exact'")
#         if self.angle < 0.0 or self.angle > 1.0:
#             raise ValueError("'angle' must be between 0.0 - 1.0")
#         if self.method == 'barnes_hut':
#             X = self._validate_data(X, accept_sparse=['csr'],
#                                     ensure_min_samples=2,
#                                     dtype=[np.float32, np.float64])
#         else:
#             X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
#                                     dtype=[np.float32, np.float64])
#         if self.metric == "precomputed":
#             if isinstance(self.init, str) and self.init == 'pca':
#                 raise ValueError("The parameter init=\"pca\" cannot be "
#                                  "used with metric=\"precomputed\".")
#             if X.shape[0] != X.shape[1]:
#                 raise ValueError("X should be a square distance matrix")
#
#
#
#             if self.method == "exact" and issparse(X):
#                 raise TypeError(
#                     'TSNE with method="exact" does not accept sparse '
#                     'precomputed distance matrix. Use method="barnes_hut" '
#                     'or provide the dense distance matrix.')
#
#         if self.method == 'barnes_hut' and self.n_components > 3:
#             raise ValueError("'n_components' should be inferior to 4 for the "
#                              "barnes_hut algorithm as it relies on "
#                              "quad-tree or oct-tree.")
#         random_state = check_random_state(self.random_state)
#
#         if self.early_exaggeration < 1.0:
#             raise ValueError("early_exaggeration must be at least 1, but is {}"
#                              .format(self.early_exaggeration))
#
#         if self.n_iter < 250:
#             raise ValueError("n_iter should be at least 250")
#
#         n_samples = X.shape[0]
#
#         neighbors_nn = None
#         if self.method == "exact":
#             # Retrieve the distance matrix, either using the precomputed one or
#             # computing it.
#             if self.metric == "precomputed":
#                 distances = X
#             else:
#                 if self.verbose:
#                     print("[t-SNE] Computing pairwise distances...")
#
#                 if self.metric == "euclidean":
#                     distances = pairwise_distances(X, metric=self.metric,
#                                                    squared=True)
#                 else:
#                     distances = pairwise_distances(X, metric=self.metric,
#                                                    n_jobs=self.n_jobs)
#
#                 if np.any(distances < 0):
#                     raise ValueError("All distances should be positive, the "
#                                      "metric given is not correct")
#
#             # compute the joint probability distribution for the input space
#             P = prob(distances, self.perplexity)
#             assert np.all(np.isfinite(P)), "All probabilities should be finite"
#             assert np.all(P >= 0), "All probabilities should be non-negative"
#             assert np.all(P <= 1), ("All probabilities should be less "
#                                     "or then equal to one")
#
#         else:
#             # Compute the number of nearest neighbors to find.
#             # LvdM uses 3 * perplexity as the number of neighbors.
#             # In the event that we have very small # of points
#             # set the neighbors to n - 1.
#             n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))
#
#             if self.verbose:
#                 print("[t-SNE] Computing {} nearest neighbors..."
#                       .format(n_neighbors))
#
#             # Find the nearest neighbors for every point
#             knn = NearestNeighbors(algorithm='auto',
#                                    n_jobs=self.n_jobs,
#                                    n_neighbors=n_neighbors,
#                                    metric=self.metric)
#             t0 = time()
#             knn.fit(X)
#             duration = time() - t0
#             if self.verbose:
#                 print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
#                     n_samples, duration))
#
#             t0 = time()
#             distances_nn = knn.kneighbors_graph(mode='distance')
#             duration = time() - t0
#             if self.verbose:
#                 print("[t-SNE] Computed neighbors for {} samples "
#                       "in {:.3f}s...".format(n_samples, duration))
#
#             # Free the memory used by the ball_tree
#             del knn
#
#             if self.metric == "euclidean":
#                 # knn return the euclidean distance but we need it squared
#                 # to be consistent with the 'exact' method. Note that the
#                 # the method was derived using the euclidean method as in the
#                 # input space. Not sure of the implication of using a different
#                 # metric.
#                 distances_nn.data **= 2
#
#             # compute the joint probability distribution for the input space
#             P = prob(distances_nn, self.perplexity)
#
#         if isinstance(self.init, np.ndarray):
#             X_embedded = self.init
#         elif self.init == 'pca':
#             pca = PCA(n_components=self.n_components, svd_solver='randomized',
#                       random_state=random_state)
#             X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
#         elif self.init == 'random':
#             # The embedding is initialized with iid samples from Gaussians with
#             # standard deviation 1e-4.
#             X_embedded = 1e-4 * random_state.randn(
#                 n_samples, self.n_components).astype(np.float32)
#         else:
#             raise ValueError("'init' must be 'pca', 'random', or "
#                              "a numpy array")
#
#         # Degrees of freedom of the Student's t-distribution. The suggestion
#         # degrees_of_freedom = n_components - 1 comes from
#         # "Learning a Parametric Embedding by Preserving Local Structure"
#         # Laurens van der Maaten, 2009.
#         degrees_of_freedom = max(self.n_components - 1, 1)
#
#         return self._tsne(P, degrees_of_freedom, n_samples,
#                           X_embedded=X_embedded,
#                           neighbors=neighbors_nn,
#                           skip_num_points=skip_num_points)

    # def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
    #           neighbors=None, skip_num_points=0):
    #     """Runs t-SNE."""
    #     # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
    #     # and the Student's t-distributions Q. The optimization algorithm that
    #     # we use is batch gradient descent with two stages:
    #     # * initial optimization with early exaggeration and momentum at 0.5
    #     # * final optimization with momentum at 0.8
    #     params = X_embedded.ravel()
    #
    #     opt_args = {
    #         "it": 0,
    #         "n_iter_check": self._N_ITER_CHECK,
    #         "min_grad_norm": self.min_grad_norm,
    #         "learning_rate": self.learning_rate,
    #         "verbose": self.verbose,
    #         "kwargs": dict(skip_num_points=skip_num_points),
    #         "args": [P, degrees_of_freedom, n_samples, self.n_components],
    #         "n_iter_without_progress": self._EXPLORATION_N_ITER,
    #         "n_iter": self._EXPLORATION_N_ITER,
    #         "momentum": 0.5,
    #     }
    #     if self.method == 'barnes_hut':
    #         obj_func = gradient_KL_divergence()
    #         opt_args['kwargs']['angle'] = self.angle
    #         # Repeat verbose argument for _kl_divergence_bh
    #         opt_args['kwargs']['verbose'] = self.verbose
    #         # Get the number of threads for gradient computation here to
    #         # avoid recomputing it at each iteration.
    #         opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
    #     else:
    #         obj_func = kl_divergence
    #
    #     # Learning schedule (part 1): do 250 iteration with lower momentum but
    #     # higher learning rate controlled via the early exaggeration parameter
    #     P *= self.early_exaggeration
    #     params, kl_divergence, it = _gradient_descent(obj_func, params,
    #                                                   **opt_args)
    #     if self.verbose:
    #         print("[t-SNE] KL divergence after %d iterations with early "
    #               "exaggeration: %f" % (it + 1, kl_divergence))
    #
    #     # Learning schedule (part 2): disable early exaggeration and finish
    #     # optimization with a higher momentum at 0.8
    #     P /= self.early_exaggeration
    #     remaining = self.n_iter - self._EXPLORATION_N_ITER
    #     if it < self._EXPLORATION_N_ITER or remaining > 0:
    #         opt_args['n_iter'] = self.n_iter
    #         opt_args['it'] = it + 1
    #         opt_args['momentum'] = 0.8
    #         opt_args['n_iter_without_progress'] = self.n_iter_without_progress
    #         params, kl_divergence, it = _gradient_descent(obj_func, params,
    #                                                       **opt_args)
    #
    #     # Save the final number of iterations
    #     self.n_iter_ = it
    #
    #     if self.verbose:
    #         print("[t-SNE] KL divergence after %d iterations: %f"
    #               % (it + 1, kl_divergence))
    #
    #     X_embedded = params.reshape(n_samples, self.n_components)
    #     self.kl_divergence_ = kl_divergence
    #
    #     return X_embedded


    # def gd(self, plot):
    #     if plot is True:
    #         plt.draw()
    #         plt.pause(2)












