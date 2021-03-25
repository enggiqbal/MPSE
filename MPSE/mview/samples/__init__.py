import os, sys
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,directory)
import csv

import numpy as np

def mnist(n_samples=1000, digits=None, **kwargs):
    from keras.datasets import mnist
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    if digits is not None:
        indices = [i for i in range(len(Y_train)) if Y_train[i] in digits]
        X_train = X_train[indices]
        Y_train = Y_train[indices]
    X = X_train[0:n_samples]
    labels = Y_train[0:n_samples]
    X = X.reshape(n_samples,28*28)
    X = np.array(X,dtype='float')/256
    return X, labels

def sload(dataset, **kwargs):
    
    data = {}
    keys = ['distances','features','sample_labels','sample_classes',
            'colors','edges',
            'labels']
    
    for key in keys:
        data[key] = None

    if dataset == 'equidistant':
        from clusters import equidistant
        data['distances'] = equidistant(**kwargs)
    elif dataset == 'clusters':
        from clusters import clusters
        data['distances'], data['sample_classes'] = clusters(**kwargs)
        data['sample_colors'] = data['sample_classes']
    elif dataset == 'clusters2':
        from clusters import clusters2
        data['features'], data['sample_classes'] = clusters2(**kwargs)
        data['distances'] = data['features']
        data['sample_colors'] = data['sample_classes']
    elif dataset == 'mnist':
        data['features'], data['sample_classes'] = mnist(**kwargs)
        data['distances'] = data['features']
        data['sample_colors'] = data['sample_classes']
    else:
        print('***dataset not found***')

    return data

def mload(dataset, n_samples=100, n_perspectives=2, **kwargs):
    "returns dictionary with datasets"

    distances = []
    data = {}
    if dataset == 'equidistant':
        length = n_samples*(n_samples-1)//2
        for persp in range(n_perspectives):      
            distances.append(np.random.normal(1,0.1,length))
        data['image_colors'] = n_samples-1
    elif dataset == 'disk':
        import misc, projections
        X = misc.disk(n_samples, dim=3)
        proj = projections.PROJ()
        Q = proj.generate(number=n_perspectives, method='random')
        Y = proj.project(Q,X)
        data['true_images'] = Y
        data['true_embedding'] = X
        data['true_projections'] = Q
        distances = Y
        data['image_colors'] = 0
    elif dataset == 'clusters2a':
        from clusters import createClusters
        D, data['image_colors'] = \
            createClusters(n_samples, n_perspectives)
    elif dataset == 'clusters':
        from clusters import clusters
        distances = []
        data['image_classes'] = []
        data['image_colors'] = []
        if 'n_clusters' in kwargs:
            n_clusters = kwargs.pop('n_clusters')
        if isinstance(n_clusters,int):
            n_clusters = [n_clusters]*n_perspectives
        else:
            n_perspectives = len(n_clusters)
        for i in range(n_perspectives):
            d, c = clusters(n_samples, n_clustesr=n_clusters[i], **kwargs)
            distances.append(d)
            data['image_classes'].append(c)
            data['image_colors'].append(c)
    elif dataset == 'clusters2':
        from clusters import clusters2
        distances = []; data['image_colors'] = []
        if 'n_clusters' in kwargs:
            n_clusters = kwargs['n_clusters']
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]*n_perspectives
        for persp in range(n_perspectives):
            d, c = clusters2(n_samples,n_clusters[persp])
            distances.append(d); data['image_colors'].append(c)
    elif dataset == '123':
        import projections
        X = np.genfromtxt(directory+'/123/123.csv',delimiter=',')
        X1 = np.genfromtxt(directory+'/123/1.csv',delimiter=',')
        X2 = np.genfromtxt(directory+'/123/2.csv',delimiter=',')
        X3 = np.genfromtxt(directory+'/123/3.csv',delimiter=',')
        proj = projections.PROJ()
        Q = proj.generate(number=3,method='cylinder')
        distances = [X1,X2,X3]
        data['true_embedding'] = X
        data['true_projections'] = Q
        data['true_images'] = [X1,X2,X3]
        data['colors'] = True
    elif dataset == 'florence':
        import florence
        distances, dictf = florence.setup()
        for key, value in dictf.items():
            data[key] = value
    elif dataset == 'credit':
        import csv
        path = directory+'/credit/'
        Y = []
        for ind in ['1','2','3']:
            filec = open(path+'discredit3_tsne_cluster_1000_'+ind+'.csv')
            array = np.array(list(csv.reader(filec)),dtype='float')
            array += np.random.randn(len(array),len(array))*1e-4
            Y.append(array)
        distances = Y
    elif dataset == 'phishing':
        import phishing
        features = phishing.features
        labels = phishing.group_names
        if n_samples is None:
            n_samples = len(features[0])
        Y, perspective_labels = [], []
        for group in [0,1,2,3]:
            assert group in [0,1,2,3]
            Y.append(features[group][0:n_samples])
            perspective_labels.append(labels[group])
        sample_colors = phishing.results[0:n_samples]
        distances = Y
        data['sample_colors'] = sample_colors
        data['perspective_labels'] = perspective_labels
    elif dataset == 'mnist':
        X, data['sample_colors'] = mnist(**kwargs)
        data['features'] = X
        distances = [X[:,0:28*14],X[:,28*14::]]
        data['sample_classes'] = data['sample_colors']
    else:
        print('***dataset not found***')
    return distances, data
