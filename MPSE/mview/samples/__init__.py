import os, sys
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,directory)
import csv

import numpy as np

def disk(n_samples=1000):
    import misc, projections
    X = misc.disk(n_samples, dim=3)
    proj = projections.PROJ()
    Q = proj.generate(number=3, method='standard')
    Y = proj.project(Q,X)
    return Y, X, Q

def clusters(n_samples=500, n_perspectives=3,**kwargs):
    from createClusters import createClusters
    a,b = createClusters(n_samples, n_perspectives)
    print(a)
    print(b)
    return a, b

def e123():
    import projections
    X = np.genfromtxt(directory+'/123/123.csv',delimiter=',')
    X1 = np.genfromtxt(directory+'/123/1.csv',delimiter=',')
    X2 = np.genfromtxt(directory+'/123/2.csv',delimiter=',')
    X3 = np.genfromtxt(directory+'/123/3.csv',delimiter=',')
    proj = projections.PROJ()
    Q = proj.generate(number=3,method='cylinder')
    return [X1,X2,X3], X, Q
    
def cluster():
    import csv
    path = directory+'/cluster/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'dist_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        Y.append(array)
    labels = open(path+'labels.csv')
    labels = np.array(list(csv.reader(labels)),dtype=int).T
    return Y, labels

def florence():
    sys.path.insert(1,directory+'/florence')
    import setup_florence as setup
    return setup.setup2()

def credit():
    import csv
    path = directory+'/credit/'
    Y = []
    for ind in ['1','2','3']:
        filec = open(path+'discredit3_tsne_cluster_1000_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')
        array += np.random.randn(len(array),len(array))*1e-4
        Y.append(array)
    return Y

def phishing(groups=[0,1,2,3], n_samples=None):
    import phishing
    features = phishing.features
    labels = phishing.group_names

    if n_samples is None:
        n_samples = len(features[0])
    Y, perspective_labels = [], []
    for group in groups:
        assert group in [0,1,2,3]
        Y.append(features[group][0:n_samples])
        perspective_labels.append(labels[group])
        
    sample_colors = phishing.results[0:n_samples]
    return Y, sample_colors, perspective_labels

def mnist0():
    Y = []
    for ind in ['1','2']:
        filec = open(directory+'/MNIST/MNIST_'+ind+'.csv')
        array = np.array(list(csv.reader(filec)),dtype='float')/256
        Y.append(array)
    filec = open(directory+'/MNIST/MNIST_labels.csv')
    labels =  np.array(list(csv.reader(filec)),dtype='float')
    labels = labels.T[0]
    filec = open(directory+'/MNIST/MNIST_labels.csv')
    X = np.array(list(csv.reader(filec)),dtype='float')/256
    return Y, labels, X

def mnist(n_samples=1000, **kwargs):
    from keras.datasets import mnist
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    X = X_train[0:n_samples]
    labels = Y_train[0:n_samples]
    X = X.reshape(n_samples,28*28)
    X = np.array(X,dtype='float')/256
    return X, labels

def load(dataset, **kwargs):
    "returns dictionary with datasets"
    datasets = ['disk','clusters','123','cluster','florence','credit','phishing','mnist',
                'mnist0']
    assert dataset in datasets
    
    data = {}
    keys = ['D','X','Q','Y','colors','embedding_colors','image_colors',
            'edges','labels']
    for key in keys:
        data[key] = None
    
    if dataset == 'disk':
        data['Y'], data['X'], data['Q'] = disk(**kwargs)
        data['D'] = data['Y']
        data['colors'] = True
    elif dataset == 'clusters':
        data['D'], data['image_colors'] = clusters(**kwargs)
    elif dataset == '123':
        data['Y'], data['X'], data['Q'] = e123(**kwargs)
        data['D'] = data['Y']
        data['colors'] = True
    elif dataset == 'cluster':
        data['D'], data['colors'] = cluster(**kwargs)
    elif dataset == 'florence':
        dictf = florence()
        data['D'] = dictf['data']
        data['labels'] = dictf['edges']
    elif dataset == 'credit':
        data['D'] = credit()
    elif dataset == 'phishing':
        data['D'], data['colors'], data['perspective_labels'] = \
            phishing(groups=[0,1,3], n_samples=200)
    elif dataset == 'mnist':
        X, data['colors'] = mnist(**kwargs)
        data['X'] = X
        data['D'] = [X[:,0:28*14],X[:,28*14::]]
        data['Q'] = 'standard'
    elif dataset == 'mnist0':
        data['D'], data['colors'], data['X'] = mnist0()
    return data

def load_single(dataset, **kwargs):
    datasets = ['clusters','mnist']
    assert dataset in datasets
    
    data = {}
    keys = ['D','X','colors','colors','edges','labels']
    
    for key in keys:
        data[key] = None
        
    if dataset == 'clusters':
        import single
        data['D'] = single.clusters(**kwargs)
    elif dataset == 'mnist':
        X, data['colors'] = mnist(**kwargs)
        data['D'] = X

    return data
