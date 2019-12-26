import PIL; from PIL import Image, ImageFont, ImageDraw
import numpy as np; import matplotlib.pyplot as plt

import image

fonts = {
    'arial' : 'fonts/arial.ttf',
    'chunkfive' : 'fonts/chunkfive.otf',
    'lilita_one' : 'fonts/LilitaOne-Regular.ttf',
    'anton' : 'fonts/Anton-Regular.ttf',
    'luckiest_guy' : 'fonts/LuckiestGuy-Regular.ttf',
    'prata' : 'fonts/Prata-Regular.ttf',
    'aldrich' : 'fonts/Aldrich-Regular.ttf',
    'shrikhand' : 'fonts/Shrikhand-Regular.ttf',
    'leckerli_one' : 'fonts/LeckerliOne-Regular.ttf',
    'ceviche_one' : 'fonts/CevicheOne-Regular.ttf',
    'fontdiner_swanky' : 'fonts/FontdinerSwanky-Regular.ttf',
    'goblin_one' : 'fonts/GoblinOne-Regular.ttf',
    'gravitas_one' : 'fonts/GravitasOne-Regular.ttf',
    'spicy_rice' : 'fonts/SpicyRice-Regular.ttf'
    }

def pil(string,font='chunkfive',points=240,crop=True,plot=False):
    """\
    Return PIL.Image of given string

    Parameters:

    string : string
    Text string to be imaged.

    font : string
    Font name. Options are arial & chunkfive.

    points : int
    Font size in points.

    crop : boolean
    Remove extra vertical space if True.

    plot : boolean
    Show plot of image if True.
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
    
    if plot is True:
        img.show()

    return img

def array(string,**kwargs):
    """\
    Returns array(s) corresponding to string(s), as given by text.pil()
    """

    if isinstance(string,list):
        return_list = True
        strings = string
    else:
        strings = [string]
        return_list = False
    
    arrays = []; 
    for string in strings:
        img = pil(string,**kwargs)
        arrays.append(np.array(img.getdata(),float).\
                      reshape(img.size[1],img.size[0]))

    if return_list is False:
        arrays = arrays[0]
        
    return arrays

if __name__=='__main__':
    #pil('xyx',plot=True)
    array(['1','2','3'],plot=True)
