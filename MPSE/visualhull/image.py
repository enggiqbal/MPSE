import os, sys
import numpy as np
import scipy as sp
from scipy import ndimage
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numbers, math, copy

import text

class Img(object):

    def __init__(self, array, atype='image', template=None,
                 dpi=1.0, shift='center', sigma=0.0, label=None):
        """
        Initializes Img object.
        """
        if atype == 'sample':
            assert template is not None

        if template is None:
            self.shape = array.shape

            if isinstance(dpi,numbers.Number):
                self.dpi = (1.0*dpi,1.0*dpi)
            elif isinstance(dpi,tuple):
                self.dpi = (1.0*dpi[0],1.0*dpi[1])
            else:
                print('Error: dpi must be number or tuple')

            if isinstance(shift,tuple):
                self.shift = (1.0*shift[0],1.0*shift[1])
            elif shift == 'center':
                self.shift = (self.shape[0]/2.0,self.shape[0]/2.0)
            else:
                print('Error: shift must be tuple or "center"')
        else:
            self.dpi = template.dpi
            self.shape = template.shape
            self.shift = template.shift
            
        if atype == 'image':
            if isinstance(array,np.ndarray):
                self.array = array
            elif isinstance(array,PIL.Image.Image):
                self.array = np.array(array.getdata(),float).\
                             reshape(array.size[1],array.size[0])
            else:
                sys.exit('Error: input array must be numpy array or PIL image')
        elif atype == 'sample':
            self.array = np.zeros(template.array.shape)
            for coord in array:
                loc = self.coord2loc(coord)
                if loc is not None:
                    self.array[loc] += 1.0

        self.sigma = sigma
        if sigma == 'optimal':
            if atype == 'sample':
                self.sigma = np.sqrt(np.sum(template.array)/len(array)/
                                     (1-np.exp(-1/2)))
            else:
                self.sigma = 0
        self.set_density()

        self.label = label

    def coord2loc(self, coord):
        vloc = math.floor(-coord[1]*self.dpi[0]+self.shift[0])
        hloc = math.floor(coord[0]*self.dpi[1]+self.shift[1])
        if 0 <= vloc < self.shape[0] and 0 <= hloc < self.shape[1]:
            return (vloc,hloc)
        else:
            return None

    def set_density(self,sigma=None):
        if sigma is not None:
            assert sigma >= 0
            self.sigma = sigma
        if self.sigma == 0:
            self.density = np.copy(self.array)###
        else:
            self.density = sp.ndimage.gaussian_filter(self.array,self.sigma)
        self.density /= np.sum(self.density)

    def plot_array(self, block=True, show=True):
        image_array = self.array
        fig,ax = plt.subplots()
        plt.gray()
        ax.imshow(image_array)
        ax.set_aspect(1.0)
        if show is True:
            plt.show(block=True)
        return fig

    def plot_density(self,block=True,show=True):
        fig = plt.figure()
        plt.gray()
        plt.imshow(self.density)
        if show is True:
            plt.show(block=block)
        return fig

    def plot(self,show=True,block=True):
        fig, axs = plt.subplots(1,2)
        plt.tight_layout()
        if self.label is not None:
            plt.suptitle(self.label)
        plt.gray()
        axs[0].imshow(self.array)
        axs[0].set_aspect(1.0)
        axs[1].imshow(self.density)
        if show is True:
            plt.show(block=block)
        
    def add_point(self, loc):
        if isinstance(loc,tuple):
            self.array[loc] += 1.0
        elif isinstance(loc,list):
            for l in loc:
                self.array[l] += 1.0
        self.set_density()

    def sample(self,N,show=False):
        blength = np.array([self.shape[1]/self.dpi[1],
                            self.shape[0]/self.dpi[0]])
        bshift = np.array([self.shift[1]/self.dpi[1],
                           self.shift[0]/self.dpi[0]])
        probs = self.density/np.max(self.density)
        
        Y = np.empty((N,2))
        n = 0
        while n < N:
            y0 = np.random.rand(3)
            y = y0[0:2]*blength-bshift
            loc = self.coord2loc(y)
            if y0[2] < probs[loc]:
                Y[n] = y
                n += 1
        if show is True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(Y[:,0],Y[:,1])
            ax.set_aspect(1.0)
            plt.show()
        return Y
        
def ddist(img1,img2,metric='l1'):
    metrics = ['l1','hellinger']
    assert metric in metrics
    mu1 = img1.density; mu2 = img2.density
    if metric == 'l1':
        dist = np.sum(np.abs(mu1-mu2))
    elif metric == 'hellinger':
        dist = np.linalg.norm(np.sqrt(mu1)-np.sqrt(mu2))/np.sqrt(2)
    return dist

def plot(imgs,smooth=True,show=True,block=True):
    """\
    Plot figure containing images in list
    """
    if isinstance(imgs,list) is False:
        imgs = [imgs]
    imgs_number = len(imgs)
    fig, axes = plt.subplots(1,imgs_number,sharey=True,
                             figsize=(2*imgs_number,3))
    plt.gray(); plt.tight_layout()
    for i in range(imgs_number):
        img = imgs[i]
        if smooth:
            array = img.density
        else:
            array = img.array
        axes[i].imshow(array)
        plt.title(img.label)
    if show is True:
        plt.show(block=block)
    return fig


##### Tests #####

def example1():
    img = text.images('0')
    Y = img.sample(100,show=True)
    np.save('0.npy',Y)

def example_sample(string='1',N=1000):
    img0 = text.images(string)
    Y = img0.sample(N,show=True)#,block=False)
    img = Img(Y,atype='sample',template=img0)

    sigmas = [0,1,2,4,6,8,10,12,16]
    imgs = []
    l1dist = np.empty(len(sigmas))
    hdist = np.empty(len(sigmas))
    for i in range(len(sigmas)):
        img1 = copy.deepcopy(img)
        img1.set_density(sigmas[i])
        l1dist[i] = ddist(img0,img1,metric='l1')
        hdist[i] = ddist(img0,img1,metric='hellinger')
        imgs.append(img1)
    plot(imgs)

    fig = plt.figure()
    plt.plot(sigmas,l1dist,label='l1')
    plt.plot(sigmas,hdist,label='hellinger')
    plt.title('Distance to Original Image')
    plt.xlabel('sigma')
    plt.legend()
    plt.show()



if __name__=='__main__':
    example1()
