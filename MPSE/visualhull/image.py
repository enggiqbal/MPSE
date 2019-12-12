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

    def __init__(self, array, dpi=1.0, justify=None, label=None):
        """
        Initializes Img object.

        Parameters:

        array : numpy array
        Array corresponding to gray-scale image in standard form, with values 
        in [0,1].

        dpi : number or tuple of numbers
        Dots-per-inch. Number of columns/rows that go into a single unit in the
        plane. If a tuple is given, the first number applies to the number of
        columns.

        justify : None or string
        If None, dpi is as specified. Otherwise, dpi is determined according to
        options 'vertical', 'horizontal', 'square'.
        """
        self.array = array 
        self.shape = array.shape
            
        if isinstance(dpi,numbers.Number):
            assert dpi > 0
            dpi = (dpi,dpi)
        else:
            assert isinstance(dpi,tuple)
            assert dpi[0] > 0 and dpi[1] > 0
        if justify is not None:
            if justify == 'vertical':
                dpi = (self.shape[0],self.shape[0])
            elif justify == 'horizontal':
                dpi = (self.shape[1],self.shape[1])
            elif justify is 'square':
                dpi = self.shape
        self.dpi = dpi
        
        self.shift = (self.shape[1]/2.0,self.shape[0]/2.0)
            
        self.label = label

    def figure(self, block=True, show=True):
        """\
        Returns imshow grayscale figure of image array.
        """
        fig,ax = plt.subplots()
        plt.gray()
        ax.imshow(self.array)
        ax.set_aspect(1.0)
        if show is True:
            plt.show(block=True)
        return fig

    def loc2coord(self, loc):
        """\
        Takes a pixel location in image array and returns a corresponding 
        coordinate in the plane.
        """
        x = (loc[1]-self.shape[1]/2.0)/self.dpi[1]
        y = (-loc[0]+self.shape[0]/2.0)/self.dpi[0]
        return (x,y)
    
    def coord2loc(self, coord):
        """\
        Takes a coordinate in the plane to the corresponding pixel location(s) 
        in the image array(s). If the coordinate(s) fall outside of the image, 
        it returns None.
        """
        vloc = math.floor(-(coord[1]*self.dpi[0]-self.shape[0]/2.0))
        hloc = math.floor(coord[0]*self.dpi[1]+self.shape[1]/2.0)
        if 0 <= vloc < self.shape[0] and 0 <= hloc < self.shape[1]:
            return (vloc,hloc)
        else:
            return None

    def belongs(self, coord, threshold=0):
        """\
        Checks if coordinate belongs to image silhoute.
        """
        loc = self.coord2loc(coord)
        if loc is None:
            return False
        else:
            value = self.array[loc]
            if value <= threshold:
                return False
            else:
                return True

    def sample(self, N, plot=False, **kwargs):
        """\
        Returns a sample of points in silhoute, uniformly sampled.
        """
        blength = np.array([self.shape[1]/self.dpi[1],
                            self.shape[0]/self.dpi[0]])
        
        X = np.empty((N,2))
        n = 0
        while n < N:
            coord = (np.random.rand(2)-0.5)*blength
            if self.belongs(coord,**kwargs):
                X[n] = coord
                n += 1
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(X[:,0],X[:,1]) 
            ax.set_aspect(1.0)
            plt.show()
        return X

def images(arrays,labels=None,**kwargs):
    """\
    Define Img objects for each array in list of arrays.
    
    Parameters:
    
    arrays : list of 2-dimensional numpy arrays
    List of arrays corresponding to images.

    labels: None or string or list of strings
    Labels for Img objects.
    
    Returns:
    
    imgs : list of Img objects
    List of Img objects corresponding to list of arrays.
    """
    imgs = []
    for i in range(len(arrays)):
        array = arrays[i]
        if labels is None or isinstance(labels,str):
            label = labels
        else:
            label = labels[i]
        imgs.append(Img(array,label=label,**kwargs))
    return imgs

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

def sample(N=100,string='0',justify='square'):
    array = text.array(string)
    img = Img(array,justify=justify)
    img.figure()
    Y = img.sample(N,plot=True)

if __name__=='__main__':
    sample(N=1000,string='b',justify='square')
