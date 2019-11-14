from autograd import grad
import math
import autograd.numpy as np
import torch.nn.functional  as F
from autograd import jacobian
from autograd import elementwise_grad
import pdb
import time, os;
from . import data;
class multiview:
    def __init__(self, D, P, dim, eps=1e-6, projection_set=1, number_of_weights=3):
        self.P1=P[0];self.P2=P[1];self.P3=P[2]
        '''
        self.P1=np.array([[1,0,0], [0,0,0],[0,0,1]], dtype=float)
        self.P2=np.array([[1,0,0], [0,1,0],[0,0,0]], dtype=float)
        self.P3=np.array([[0,0,0], [0,1,0],[0,0,1]], dtype=float)
        if projection_set==2:
            self.P1=np.array([[1,0,0], [0,0,0],[0,0,1]], dtype=float)
            self.P2=np.array([[1/4,np.sqrt(3)/4,0], [np.sqrt(3)/4,3/4,0],[0,0,1]], dtype=float)
            self.P3=np.array([[1/4,-1*np.sqrt(3)/4,0], [-1*np.sqrt(3)/4,3/4,0],[0,0,1]], dtype=float)

            #self.P1=np.array([[1,0,0], [0,0,0],[0,0,1]], dtype=float)
            #self.P2=np.array([[1/2,-1*np.sqrt(3)/2,0], [np.sqrt(3)/2,1/2,0],[0,0,1]], dtype=float)
            #self.P3=np.array([[-0.5,-1*np.sqrt(3)/2,0], [np.sqrt(3)/2,1/2,0],[0,0,1]], dtype=float)

'''
        self.D1=D[0]
        self.D2=D[1]
        self.D3= D[2]
        self.dim=dim
        self.eps=eps
        self.number_of_weights = number_of_weights
    def getparameters(self):
        return self.D1 , self.D2, self.D3, self.P1, self.P2, self.P3, self.dim
    def get_projection_point(self, P,X):
        t=(-P[3] - X[0]*P[0] -  X[1]*P[1] - X[2]*P[2]) / (P[0] * P[0] +P[1] * P[1]+P[2] * P[2])
        vi=np.array([P[0] * t + X[0], P[1] * t + X[1], P[2] * t + X[2]])
        return vi

    def costfunction_projection(self,A,P1,P2,P3):
        D1, D2, D3, _, _, _, dim=self.getparameters()
        m=int(len(A)/dim)
        X=A.reshape(m,dim)
        cost=0
        eps=self.eps
        m=len(X)
        for i in range(0, m):
            for j in range(i+1, m):

                vi=self.get_projection_point(P1,X[i])
                vj=self.get_projection_point(P1,X[j])
                c1=np.square(np.sqrt(np.sum(np.array( np.square( vi-vj)))) - D1[i][j])

                #diff=np.sum( np.square( vi-vj))
                #d= np.sqrt(diff)
                #cost=np.sum([cost, np.square(d - D1[i][j])]) #if  np.abs(d - D1[i][j]) > eps else cost
                vi=self.get_projection_point(P2,X[i])
                vj=self.get_projection_point(P2,X[j])
                c2=np.square(np.sqrt(np.sum(np.array( np.square( vi-vj)))) - D2[i][j])

                #diff=np.sum( np.square( vi-vj))
                #d= np.sqrt(diff)
                #cost=np.sum([cost, np.square(d - D2[i][j])])# if  np.abs(d - D2[i][j]) > eps else cost
                vi=self.get_projection_point(P3,X[i])
                vj=self.get_projection_point(P3,X[j])
                c3=np.square(np.sqrt(np.sum(np.array( np.square( vi-vj)))) - D3[i][j])

                #diff=np.sum( np.square( vi-vj))
                #d= np.sqrt(diff)
                #cost=np.sum([cost, np.square(d - D3[i][j])]) #if  np.abs(d - D3[i][j]) > eps else cost
                cost=cost+c1+c2+c3
        return cost

    def costfunction(self,A):
        D1, D2, D3, P1, P2, P3, dim=self.getparameters()
        m=int(len(A)/dim)
        X=A.reshape(m,dim)
        cost=0
        #cost=self.projectionCost(X,P1, D1 )+ self.projectionCost( X,P2, D2 ) #+self.projectionCost( X,P3, D3 )

#        X, P, W=X,P1, D1
        eps=self.eps
        m=len(X)
        for i in range(0, m):
            for j in range(i+1, m):
                vi=P1 @ X[i]
                vj=P1 @ X[j]
                diff=np.sum( np.square( vi-vj))
                d= np.sqrt(diff)
                cost=np.sum([cost, np.square(d - D1[i][j])]) #if  np.abs(d - D1[i][j]) > eps else cost
                if  self.number_of_weights>2:
                    vi=P2 @ X[i]
                    vj=P2 @ X[j]
                    diff=np.sum( np.square( vi-vj))
                    d= np.sqrt(diff)
                    cost=np.sum([cost, np.square(d - D2[i][j])])# if  np.abs(d - D2[i][j]) > eps else cost
                if self.number_of_weights==3:
                    vi=P3 @ X[i]
                    vj=P3 @ X[j]
                    diff=np.sum( np.square( vi-vj))
                    d= np.sqrt(diff)
                    cost=np.sum([cost, np.square(d - D3[i][j])]) #if  np.abs(d - D3[i][j]) > eps else cost

        return cost
    def projectionCost(self, X, P, W):
        eps=self.eps
        cost=0
        m=len(X)
        for i in range(0, m):
            for j in range(i+1, m):
                vi=P @ X[i]
                vj=P @ X[j]
                diff=np.sum( np.square( vi-vj))
                d=0 if diff<eps else np.sqrt(diff)
                cost=cost+ np.square(d - W[i][j]) if  abs(d - W[i][j]) > eps else cost
            return cost



    def multiview_mds(self, A,steps, alpha, stopping_eps,outputpath,name_data_set, save_progress=0, verbose=1):
        g=jacobian(self.costfunction)#grad(self.costfunction)
        js_file_path=os.path.join(outputpath, name_data_set +"_coordinates_tmp.js")

        costs=[]
        costs.append(self.costfunction(A))
        for i in range(1,steps+1):
            A=A- alpha * g(A)
            newcost= self.costfunction(A)
            if verbose:
                print("step: ", i, ", cost:", newcost)
            if save_progress:
                data.js_data_writer(A,js_file_path,costs, self.P1, self.P2, self.P3)
            if costs[i-1]-newcost < stopping_eps:
                print("early stopping at", i )
                costs.append(newcost)
                return A, costs,  [self.P1, self.P2, self.P3]
            costs.append(newcost)
        return A, costs, [self.P1, self.P2, self.P3]




    def multiview_mds_projection(self, A,P1,P2,P3,steps, alpha, stopping_eps,outputpath,name_data_set):
        #pdb.set_trace()
        js_file_path=name_data_set +"_coordinates_tmp.js"
        self.create_viz_file(outputpath,name_data_set,js_file_path)
        g=jacobian(self.costfunction_projection)#grad(self.costfunction)
        costs=[]
        costs.append(self.costfunction_projection(A,P1,P2,P3))
        proj=""
        updown=0
        for i in range(1,steps+1):
            nabla_g = elementwise_grad(self.costfunction_projection,(0,1,2,3))
            dA,dP1,dP2,dP3=nabla_g(A,P1,P2,P3)
            A=A- alpha * dA
            P1=P1- (alpha/10) * dP1
            P2=P2- (alpha/10) * dP2
            P3=P3- (alpha/10) * dP3
            newcost= self.costfunction_projection(A,P1,P2,P3)
            print(" step: ", i, ", cost:", newcost)
            self.temp_data_writer(A,outputpath+js_file_path,costs, P1, P2, P3 ,i, newcost)
            if costs[i-1]-newcost < stopping_eps:
                updown=updown+1
            else:
                updown=0
            costs.append(newcost)
            if updown==3:
                print("early stopping at", i )
                return A, costs, P1, P2, P3
            costs.append(newcost)
        return A, costs, P1, P2, P3
