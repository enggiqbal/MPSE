import pickle
import numpy as np
import math, numbers
import matplotlib.pyplot as plt
import text

import image

import sys
sys.path.insert(1,'../mview')
import perspective

class Hull(object):

    def __init__(self,imgs,persp,boundary='box',size=1.0):
        
        assert isinstance(imgs,list)
        self.K = len(imgs)
        self.imgs = imgs
        self.persp = persp
        self.dim = persp.dimX

        assert boundary in ['box']
        assert isinstance(size,numbers.Number) and size>0
        self.boundary = boundary; self.size = size
        if self.boundary is 'box':
            self.random = random_box(self.dim,self.size)

        self.N = 0
        self.X = np.empty((0,self.dim))
        self.Y = np.empty((self.K,0,2)) ###dimY

    def update_density(self):
        """\
        Compute density of each projection
        """
        import scipy.stats; from scipy.stats import gaussian_kde
        print('Density stats:')
        print('number:',self.N)
        kernel = []
        density = np.empty((self.K,self.N))
        for k in range(self.K):
            kernel.append(gaussian_kde(self.Y[k].T,bw_method=self.N**(-1/6)*.7))
            density[k] = kernel[k](self.Y[k].T)
            print(f'perspective {k} -')
            print('minimum :',np.min(density[k]))
            print('maximum :',np.max(density[k]))
            print('mean :',np.average(density[k]))
            print('stdev :',np.std(density[k]))
        self.kernel = kernel
        self.density = density

    def add_points(self,N,forgive=[]):

        assert isinstance(N,int) and N>0
        self.N += N
        self.X

        X = np.empty((N,self.dim))
        Y = np.empty((self.K,N,2))
        n = 0; it = 0; max_its = 1e6
        while n < N and it < max_its:
            x = self.random()
            y_list = self.persp.compute_Y(x) ###
            miss_num=0; k=0
            while miss_num <= len(forgive) and k<self.K:
                img = self.imgs[k]
                point = img.coord2loc(y_list[k])
                if point is None:
                    miss_num = len(forgive)+1
                elif img.array[point] == 0:
                    miss_num += 1
                k += 1
            if miss_num <= len(forgive):
                if miss_num == 0 or np.random.rand() < forgive[miss_num-1]:
                    X[n] = x
                    Y[:,n,:] = np.array(y_list)
                    n += 1
            it += 1
        self.X = np.concatenate((self.X,X),axis=0)
        self.Y = np.concatenate((self.Y,Y),axis=1)
        self.update_density()

    def compute_probs(self,bar=0.01):
        """
        Compute probability
        """
        num = math.floor(self.N*bar)
        probs = np.empty((self.K,self.N))
        for k in range(self.K):
            idx = np.argpartition(self.density[k],num)
            #print(idx[num],self.density[k][idx)
            density_bar = self.density[k][idx[num]]
            probs[k] = density_bar/self.density[k]
        return probs
        
    def remove_points(self,method='average',**kwargs):
        """\
        Remove points to uniformalize projections
        """
        probs = self.compute_probs(**kwargs)
        average = np.average(probs,axis=0)
        indices = []
        for n in range(self.N):
            if np.random.rand() < average[n]:
                indices.append(n)
        self.X = self.X[indices]
        self.Y = self.Y[:,indices]
        self.N = len(self.X)
        self.update_density()

    def figure(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        fig1 = plt.figure()
        plt.title('X')
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.X[:,0],self.X[:,1],self.X[:,2])

        fig2, axs = plt.subplots(1,3,sharex=True)
        plt.tight_layout()
        for k in range(self.K):
            axs[k].scatter(self.Y[k,:,0],self.Y[k,:,1])
            axs[k].set_aspect(1.0)
            axs[k].set_title(f'Projection {k}')
        plt.suptitle('Projections of X')
        plt.show()
        
    def figure2(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        from scipy import stats

        fig = plt.figure(figsize=(4*(self.K+1),4))
        fig.suptitle(f'Hull sample w/ projections (N = {self.N})')

        ax = fig.add_subplot(1,self.K+1,1,projection='3d')
        ax.set_title('X')
        ax.scatter3D(self.X[:,0],self.X[:,1],self.X[:,2])
        
        for k in range(self.K):  
            xmin = self.Y[k,:,0].min()
            xmax = self.Y[k,:,0].max()
            ymin = self.Y[k,:,1].min()
            ymax = self.Y[k,:,1].max()
            X, Y = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
            positions = np.vstack([X.ravel(),Y.ravel()])
            Z = np.reshape(self.kernel[k](positions).T,X.shape)

            ax = fig.add_subplot(1,self.K+1,k+2)
            ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                          extent=[xmin, xmax, ymin, ymax])
            ax.scatter(self.Y[k,:,0],self.Y[k,:,1])
            #fig2.colorbar(np.rot90(Z)[3])
            ax.set_aspect(1.0)
            ax.set_title(f'Projection {k+1}')
        plt.show()

    def save(self,filename):
        with open('hull/'+filename+'.pickle','wb') as handle:
            pickle.dump(self, handle)
        
### Boundary functions ###

def random_box(dimX,size):
    random = lambda : size*(np.random.rand(dimX)-0.5)
    return random

### Functions to add points ###

def convexhull(num,imgs,projs,box_length=1):
    K = len(imgs)
    X = np.empty((num,3))
    n = 0; it = 0; max_its = 1e6
    while n < num and it < max_its:
        x = box_length*(np.random.rand(3)-0.5)
        condition = True; k=0
        while condition is True and k<K:
            img = imgs[k]
            y = projs[k](x)
            point = img.coord2loc(y)
            if point is None:
                condition = False
            elif img.array[point] == 0:
                condition = False
            k += 1
        if condition is True:
            X[n] = x
            n += 1
        it += 1
    return X

def forgiving(num,imgs,projs,box_length=1,prob=[0.1]):
    K = len(imgs)
    X = np.empty((num,3))
    n = 0; it = 0; max_its = 1e6
    while n < num and it < max_its:
        x = box_length*(np.random.rand(3)-0.5)
        y_list = projs.project(x)
        miss_num=0; k=0
        while miss_num <= len(prob) and k<K:
            img = imgs[k]
            y = y_list[k]
            point = img.coord2loc(y)
            if point is None:
                miss_num = len(prob)+1
            elif img.array[point] == 0:
                miss_num += 1
            k += 1
        if miss_num <= len(prob):
            if miss_num == 0 or np.random.rand() < prob[miss_num-1]:
                X[n] = x
                n += 1
        it += 1
    return X

def uniform(num,imgs,projs,box_length=1,sigma=1.0):
    K = len(imgs)
    X = np.zeros((num,3))
    X0 = convexhull(math.floor(num/10),imgs,projs,
                                           box_length=1)
    X[0:math.floor(num/10.0)] = X0
    simgs = [] #sample imgs
    for k in range(K):
        Y0 = projs[k](X0)
        simgs.append(image.Img(Y0,atype='sample',template=imgs[k],
                               sigma=sigma))
    n = math.floor(num/10.0); it = 0; max_its = 1e6
    while n < num and it < max_its:
        x = box_length*(np.random.rand(3)-0.5)
        inrange=True; k=0; prob_diff=0; locs = []
        while inrange is True and k<K:
            img = imgs[k]
            simg = simgs[k]
            y = projs[k](x)
            loc = img.coord2loc(y)
            locs.append(loc)
            if loc is None:
                inrange = False
            else:
                temp = img.density[loc] - simg.density[loc]
                if temp >= 0:
                    prob_diff += temp
                else:
                    prob_diff += 2*temp
                    k += 1
        if inrange is True and prob_diff > 1.0/(2*math.pi*sigma*2*n)/5:
                X[n] = x
                for k in range(K):
                    simgs[k].add_point(locs[k])
                n += 1
        it += 1
    return X

### Tests ### CHECK: X should be filled everytime, or cut 

def example(num=1000,font='lilita_one'):
    #strings = ['1']
    strings = ['1','2','3']
    arrays = text.array(strings,font=font)
    imgs = image.images(arrays,labels=strings,justify='vertical')

    persp = perspective.Persp()
    persp.fix_Q(special='cylinder',number=len(strings))

    hull = Hull(imgs,persp)
    hull.add_points(num)
    hull.figure2()

#    for k in range(len(imgs)):
 #       Y = projs[k](X)
 #       img = image.Img(Y,atype='sample',template=imgs[k],sigma=15.0)

def example2(num=1000,font='lilita_one'):
    #strings= ['1']
    strings = ['1','2','3']
    arrays = text.array(strings,font=font)
    imgs = image.images(arrays,labels=strings,justify='vertical')
    
    persp = perspective.Persp()
    persp.fix_Q(special='cylinder',number=len(strings))

    hull = Hull(imgs,persp)
    hull.add_points(num)
    for i in range(10):
        hull.figure2()
        hull.remove_points()

    hull.figure2()
        
def example_123(num=100,forgive=[.05,.01],save_data=False):
    strings = ['1','2','3']
    imgs0 = text.images(strings,justify='vertical')
    proj = projections.Proj()
    proj.set_params_list(number=3,special='cylinder')
    hull = Hull(imgs0,proj)
    hull.add_points(num, forgive)
    hull.return_figure(); plt.show()
    if save_data:
        np.save('examples/123/true123.npy',hull.X)
        np.save('examples/123/true1.npy',hull.Y[0])
        np.save('examples/123/true2.npy',hull.Y[1])
        np.save('examples/123/true3.npy',hull.Y[2])
        np.save('examples/123/params.npy',proj.params_list)

def example_xyz(save_data=False):
    strings = ['x','y','z']
    imgs0 = text.images(strings,justify='square',font=arial)
    proj = projections.Proj(); proj.initialize(special='standard')
    hull = Hull(imgs0,proj)
    hull.add_points(1000,forgive=[.001,.0005])
    hull.return_figure()
    plt.show()

if __name__=='__main__':

    font = 'spicy_rice'
    example(font=font)
    example2(font=font)
