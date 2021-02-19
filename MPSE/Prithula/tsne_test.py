import scipy
import numpy as np
import tsne
import pandas as pd

from sklearn.datasets import load_digits
from scipy import spatial

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

def tsne_test(**kwargs):
    print("start the function")
    projection=np.random.rand(2, 3)
    print(projection)
    print(projection.shape)
    # x_true = np.load('examples/123/true123.npy')
    # x_true = pd.read_csv('input/spicy_rice_1000_123.csv', header=None)
    # dim_1=pd.read_csv('input/spicy_rice_1000_1.csv', header=None)
    # dim_2 = pd.read_csv('input/spicy_rice_1000_2.csv', header=None)
    # dim_3 = pd.read_csv('input/spicy_rice_1000_3.csv', header=None)

    #clulster dataset:
    x_true = pd.read_csv('input/spicy_rice_1000_123.csv', header=None)


    dim_1 = pd.read_csv('clusters_dataset/data_mat_1.csv', header=None)
    dim_2 = pd.read_csv('clusters_dataset/data_mat_2.csv', header=None)
    dim_3 = pd.read_csv('clusters_dataset/data_mat_3.csv', header=None)

    distance_1, distance_2, distance_3=tsne.projection(x_true, dim_1, dim_2, dim_3)
    # D = spatial.distance_matrix(x_true, x_true)
    # print(D.shape)
    # print((D[0][0]))
    # print(D[0][1])
    # print(x_true.shape)
    # print(x_true.values.shape)
    # print(x_true)
    print(x_true.values)

    # D = spatial.distance_matrix(x_true, x_true)
    # print(D)
    # X_embedded = tsne.fit(D

    # digits = load_digits()

    # Display the first digit
    # plt.figure(1, figsize=(8, 8))
    # plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()
    # X, y = load_digits(return_X_y=True)
    # print(X.shape)
    # print(y.shape)
    # print(y)
    #import pdb;pdb.set_trace()
    # X_embedded = tsne.fit(x_true)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    X_embedded = tsne.fit_new(x_true, dim_1, dim_2, dim_3)
    # X_embedded = tsne.fit_n(x_true, distance_1, distance_2, distance_3)

    # tsn = TSNE()
    # X_embedded = tsn.fit_transform(x_true)

    # print(X_embedded.shape)
    # print(X_embedded)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], legend='full', palette=palette)
    # plt.show()
    plt.figure()
    ax=plt.axes(projection="3d")
    ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],color='red',marker ='^')
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')
    plt.show()
    print(X_embedded.shape)

    # tsn = TSNE()
    # x_embedded = tsn.fit_transform(x_true)
    # plt.figure()
    # plt.plot(x_embedded[:, 0], x_embedded[:, 1],'o')

    # print(X_embedded[:,0])
    # sns.scatterplot(x_embedded[:, 0], x_embedded[:, 1], legend='full', palette=palette)
    # plt.show()
tsne_test()