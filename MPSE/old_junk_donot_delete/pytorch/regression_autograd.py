import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad
import math


alpha=0.0001

def data():
    X=np.array([[1,10,20,30,40,50], [1,40,50,60,70,80],[1,10,11,12,13,14],[1,15,21,42,53,64]])
    theta=np.array([[1.0,2.0,3.0,4.0,5.0,6.0]])
    y=np.array([[500],[600],[400],[450]])
    return X,theta,y
X,theta,y=data()


def reg_autograd(alpha,X,theta,y):
    """
    cost function of logistic regression
    """
    g=grad(func)
    for i in range(0,20):
        theta=theta- alpha * g(theta)
        print("cost", func(theta))
    print("final theta",theta)



def func(theta):
    global X
    global y
    h=X @ theta.T
    cost1=np.square(h-y)
    cost=np.sum( cost1) /(2 * len(X))
    return cost



def reg(alpha,X,theta,y):
  for i in range(0,20):
    cost1=np.square((X @ theta.T)-y)
    cost=np.sum( cost1) /(2 * len(X))
    print("cost",cost)
    d= np.sum(X * (X @ theta.T - y), axis=0)/len(X)
    #print(d)
    theta = theta - alpha * d
  print("final theta",theta)
  return theta


reg_autograd(alpha,X,theta,y)
X,theta,y=data()
reg(alpha,X,theta,y)
