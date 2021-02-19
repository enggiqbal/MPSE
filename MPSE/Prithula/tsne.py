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

