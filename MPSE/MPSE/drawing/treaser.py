import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys
import pdb


proj=""
points=""
costhistory=""
def processfile(js_file_name):
    outdir="pngs/"
    indir="../html3Dviz/"
    global proj

    #def processfile(js_file_name, outdir,indir ):
        #js_file_name="../html3Dviz/sculp_123p2_coordinates_tmp.js"
        #js_file_name="circ_tri_p1_coordinates_tmp.js"
    f=open(indir+js_file_name,"r")
    data=f.read()
    lines=data.split("\n")
    #print(len(lines))

    proj_txt=lines[4]+lines[5]+lines[6]+lines[7]+lines[8]+lines[9]+lines[10]
    proj_txt=proj_txt.replace("var","")
    proj_txt=proj_txt.replace("\n","")
    proj_txt=proj_txt.replace(";","")
    proj_txt=proj_txt.replace(" ","")
    #pdb.set_trace()
    exec(proj_txt, globals())
    points_txt=lines[11].replace("var","").replace(" ","")
    exec(points_txt, globals())
    points_data=np.array(points)
    costhistory_txt= lines[3].replace("var","").replace(" ","")
    exec(costhistory_txt, globals())
    costhistory_data=np.array(costhistory)
    return points_data

def gen_treaser(az,el,points1):
    X,Y,Z=points1.T[0], points1.T[1], points1.T[2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #plt.axis('off')
    #ax.grid(False)
#    el=-170
#    az=-100
    ax.view_init(el, az)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_zlabel('Z')
    #ax.set_proj(elev=-170, az=-100)
    ax.scatter3D(X, Y, Z, c='green', cmap='Greens')
    plt.title("3D View")
    plt.savefig( str(az)+str(el)+".png")
    plt.show()




import pandas as pd

points1=processfile("123_123p2_coordinates_tmp.js")


gen_treaser(-30, -170,points1 )
gen_treaser(-50, -170,points1 )
gen_treaser(-100, -170,points1 )
gen_treaser(-120, -170,points1 )
gen_treaser(-130, -170,points1 )
