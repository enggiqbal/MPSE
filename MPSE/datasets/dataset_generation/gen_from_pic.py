from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
def processfile(file_name,out):
#    file_name="mouse.png"
    im = Image.open(file_name)
    rgb_im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    n=1000
    points=np.array([])
    for i in range(0,im.size[0]):
        for j in range(0, im.size[1]):
            r, g, b = rgb_im.getpixel((i, j))
            if r!=0 and g!=0 and b!=0:
                points=np.append(points,[i,j])
                draw.point((i,j), fill='red')
    d=points.reshape(int(len(points)/2), 2)
    idx = np.random.randint(int(len(points)/2), size=n)
    d=d[idx,:]
    M=euclidean_distances(d)
    np.savetxt( out,M, delimiter=",")
    #im.save('result.png')
processfile("mouse.png", '../dataset_3D/cup_mouse/mouse.csv')
processfile("cup.png", '../dataset_3D/cup_mouse/cup.csv')
