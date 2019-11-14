import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys


def get2Dproject(points, P):
    svd=np.linalg.svd(P)
    proj=svd[2]
    proj=proj[:,0:2]
    return np.matmul(points, proj)


def get2dpoints(points, P):
    points2d= np.zeros((len(points),3))
    for i in range(0, len(points)):
        x=P*points[i]
        x=np.sum(x, axis=0)
        points2d[i]=x
    return points2d

def save_figure(points,js_file_name, proj, outdir):
    filename=outdir+"3d/"+js_file_name.replace(".js", proj) + ".png"
    X,Y,Z=points.T[0], points.T[1], points.T[2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(X, Y, Z, c='red', cmap='Greens')
    plt.title("3D View")
    plt.savefig(filename)
    #plt.show()
def draw2d(points,outdir,js_file_name,proj):
    filename=outdir+js_file_name.replace(".js", proj) + ".png"
    fig = plt.figure()
    ax = plt.axes()
    ax.grid(False)
    plt.axis('off')
    ax.scatter(points.T[0], points.T[1],  c='green', marker="x" ,  cmap='Greens');
    plt.savefig(filename)



outdir="pngs/"
indir="../html3Dviz/"
js_file_name="sculp_123p1_coordinates_tmp.js"

js_file_name="sculp_12p1_coordinates_tmp.js"
js_file_name="sculp_23p1_coordinates_tmp.js"
js_file_name="sculp_12p1_coordinates_tmp.js"
js_file_name=sys.argv[1]

#def processfile(js_file_name, outdir,indir ):
    #js_file_name="../html3Dviz/sculp_123p2_coordinates_tmp.js"
    #js_file_name="circ_tri_p1_coordinates_tmp.js"
f=open(indir+js_file_name,"r")
data=f.read()
lines=data.split("\n")
print(len(lines))

proj=lines[4]+lines[5]+lines[6]+lines[7]+lines[8]+lines[9]+lines[10]
proj=proj.replace("var","")
proj=proj.replace("\n","")
proj=proj.replace(";","")
proj=proj.replace(" ","")
exec(proj)
proj=np.array(proj)
print(proj.shape)

points=lines[11].replace("var","").replace(" ","")
exec(points)
points=np.array(points)

costhistory= lines[3].replace("var","").replace(" ","")
exec(costhistory)
costhistory=np.array(costhistory)


ps1=get2Dproject(points, proj[0])
ps2=get2Dproject(points, proj[1])
ps3=get2Dproject(points, proj[2])

#
#points1=get2dpoints(points, proj[0])
#save_figure(points1,js_file_name, "projection1",outdir)

#points1=get2dpoints(points, proj[1])
#save_figure(points1,js_file_name, "projection2",outdir)

#points1=get2dpoints(points, proj[2])
#save_figure(points1,js_file_name, "projection3",outdir)

#draw2d(points,1,2,outdir,js_file_name,"projection2")
#draw2d(points,2,0,outdir,js_file_name,"projection3")

draw2d(ps1,outdir,js_file_name,"projection1")
draw2d(ps2,outdir,js_file_name,"projection2")
draw2d(ps3,outdir,js_file_name,"projection3")



x=np.arange(len(costhistory))
fig = plt.figure()
ax = plt.axes()
#ax.scatter(x.T, costhistory.T,  c='green', marker="x" ,  cmap='Greens');
plt.plot(x,costhistory )
plt.savefig(outdir+ "cost"+ js_file_name.replace(".js",".png"))




outdir="pngs/"
indir="../html3Dviz/"
js_file_names=["sculp_123p1_coordinates_tmp.js"]

js_file_names=["sculp_123p1_coordinates_tmp.js"]

#for js_file_name in js_file_names:
#    processfile(js_file_name, outdir,indir )

'''




X,Y,Z=points1.T[0], points1.T[1], points1.T[2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', cmap='Greens')
plt.title("3D View")
plt.show()
plt.savefig('books_read.png')


'''
