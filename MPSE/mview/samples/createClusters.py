import numpy as np
import matplotlib.pyplot as plt
import math

def createClusters(numbPoints, numbPerspectives):
        
    retClusters = []
    labels = []

    for i in range(numbPerspectives):

        meanFirst = (0, 0)
        A = np.random.rand(2, 2)
        covFirst = [[0.5, -0.1], [-0.1, 0.5]] + np.dot(A, A.transpose())

        x = np.random.multivariate_normal(meanFirst, covFirst, numbPoints//2)
    
        meanSecond = (8 + np.random.randn(1)[0], 8 + np.random.randn(1)[0])
        A = np.random.rand(2, 2)
        covSecond = [[0.5, 0.1], [0.1, 0.5]] + np.dot(A, A.transpose())

        y = np.random.multivariate_normal(meanSecond, covSecond,
                                          (numbPoints - numbPoints//2))
        
        z = np.concatenate((x, y))
        perm = np.random.permutation(numbPoints)
        z = z[perm]

        retClusters.append(z)
        currlabels = [0]*numbPoints
        for i in range(numbPoints):
            if perm[i] >= numbPoints/2:
                currlabels[i] = 1
        print(currlabels)
        labels.append(currlabels)

    return retClusters, labels
