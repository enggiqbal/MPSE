import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'../mview')
import perspective
import image, hull, text

def one_two_three():
    strings = ['1','2','3']
    persp = perspective.Persp()
    persp.fix_Q(special='cylinder',number=3)
    for font in text.fonts:
        arrays = text.array(strings,font=font)
        imgs = image.images(arrays,labels=strings,justify='vertical')
        h = hull.Hull(imgs,persp)
        h.add_points(10000)
        np.savetxt('examples/123/'+font+'_10000_123.csv',h.X,delimiter=',')
        np.savetxt('examples/123/'+font+'_10000_1.csv',h.Y[0],delimiter=',')
        np.savetxt('examples/123/'+font+'_10000_2.csv',h.Y[1],delimiter=',')
        np.savetxt('examples/123/'+font+'_10000_3.csv',h.Y[2],delimiter=',')
        h.figure(plot=False)
        plt.savefig('examples/123/'+font+'_10000')

        for target in [1000,500,200,100]:
            h.uniformize(target=target)
            target = str(target)
            np.savetxt('examples/123/'+font+'_'+target+'_123.csv',h.X,
                       delimiter=',')
            np.savetxt('examples/123/'+font+'_'+target+'_1.csv',h.Y[0],
                       delimiter=',')
            np.savetxt('examples/123/'+font+'_'+target+'_2.csv',h.Y[1],
                       delimiter=',')
            np.savetxt('examples/123/'+font+'_'+target+'_3.csv',h.Y[2],
                       delimiter=',')
            h.figure(plot=False)
            plt.savefig('examples/123/'+font+'_'+target+'_image')

def xyz():
    strings = ['x','y','z']
    persp = perspective.Persp()
    persp.fix_Q(special='standard',number=3)
    for font in text.fonts:
        arrays = text.array(strings,font=font)
        imgs = image.images(arrays,labels=strings,justify='vertical')
        h = hull.Hull(imgs,persp)
        h.add_points(10000)
        np.savetxt('examples/xyz/'+font+'_10000_xyz.csv',h.X,delimiter=',')
        h.figure(plot=False)
        plt.savefig('examples/xyz/'+font+'_10000')

        for target in [1000,500,200,100]:
            h.uniformize(target=target)
            target = str(target)
            np.savetxt('examples/xyz/'+font+'_'+target+'_xyz.csv',h.X,
                       delimiter=',')
            h.figure(plot=False)
            plt.savefig('examples/xyz/'+font+'_'+target+'_image')

xyz()
