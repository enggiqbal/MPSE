import PIL; from PIL import Image, ImageFont, ImageDraw
import numpy as np; import matplotlib.pyplot as plt

import image

fonts = {
    'arial' : 'fonts/arial.ttf',
    'chunkfive' : 'fonts/chunkfive.otf'
    }

def pil(string,font='chunkfive',points=240,crop=True,save=False,
                show=False):
    """\
    Return PIL.Image of given string

    --- arguments ---
    string = string to be imaged
    font = font name
    points = font size in points
    crop = remove extra vertical space if True
    save = save as .png file if True
    feedback = show image if True
    """
    assert font in fonts
    Font = PIL.ImageFont.truetype(fonts[font],points)
    
    img = PIL.Image.new("1", (2*points*len(string),2*points), 0)
    draw = PIL.ImageDraw.Draw(img)
    draw.text((0,int(points/2)),string,1,font=Font)
    draw = PIL.ImageDraw.Draw(img)
    
    bbox = img.getbbox()
    if crop is False:
        img = img.crop((bbox[0],0,bbox[2],2*points))
    else:
        img = img.crop(bbox)

    if save is True:
        img.save("text_images/"+font+str(points)+"_"+string+".png")
    
    if show is True:
        img.show()

    return img

def images(strings,font='chunkfive',points=240,crop=True,justify=None):

    if isinstance(strings,list):
        return_list = True
    else:
        assert isinstance(strings,str)
        strings = [strings]
        return_list = False
    
    images = []; 
    for i in range(len(strings)):
        img = pil(strings[i],font,points,crop)
        array = np.array(img.getdata(),float).\
                reshape(img.size[1],img.size[0])

        if justify is None:
            dpi = 1.0
        elif justify is 'vertical':
            dpi = len(array)
        elif justify is 'horizonatal':
            dpi = array.shape[1]
        elif justify is 'square':
            dpi = array.shape
                
        img = image.Img(array,atype='image',dpi=dpi,label=strings[i])
        images.append(img)

    if return_list is False:
        images = images[0]
        
    return images
