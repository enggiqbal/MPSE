import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys
import pdb
from sklearn.metrics.pairwise import euclidean_distances

def get2Dproject(points, P,W):
    Z=np.min( np.sum(P,axis=1))
    A= np.sum(P,axis=1) !=Z
    svd=np.linalg.svd(P)
    #print("svd1:",svd[1])
    proj=svd[2]
    #print("projection",P)
    #print("projection svd", proj)
    #proj=proj[:,0:2]
    proj=proj[0:2,:]
    proj = proj.transpose()
    #proj=proj[:,W]
#    print("projection--->", proj)
    return np.matmul(points, proj)

def get_view_point(P):
    svd=np.linalg.svd(P)
    proj=svd[2]
    view_point=proj[2,:]
    return view_point



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

proj=""
points=""
costhistory=""
def processfile(js_file_name):
 
    f=open(js_file_name,"r")
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

    P=np.array(proj)
    #print(P.shape)
    #print(P)
    points_txt=lines[11].replace("var","").replace(" ","")
    exec(points_txt, globals())
    points_data=np.array(points)
    #print(points_data.shape)
 

    ps1=get2Dproject(points_data, P[0],[True, True, False])
    ps2=get2Dproject(points_data, P[1],[False, True, True])
    ps3=get2Dproject(points_data, P[2],[True, False, True])

    return ps1, ps2, ps3
     

def find_distortion(a,b):
    a=a.values
 
    diff=a-b
    
    return np.sum(diff ** 2)

#python3 project_distance.py ../dataset_tabluar/dissimple1000_1.csv ../dataset_tabluar/dissimple1000_2.csv ../dataset_tabluar/dissimple1000_3.csv ../html3Dviz/simple1000_123p1_coordinates_tmp.js 
#python3 project_distance.py ../dataset_tabluar/discredit2_1000_1.csv ../dataset_tabluar/discredit2_1000_2.csv ../dataset_tabluar/discredit2_1000_3.csv ../html3Dviz/simple1000_123p1_coordinates_tmp.js 
#python3 project_distance.py ../dataset_3D/123_dataset_new/250/data_mat_1_250.csv ../dataset_3D/123_dataset_new/250/data_mat_2_250.csv ../dataset_3D/123_dataset_new/250/data_mat_3_250.csv ../html3Dviz/123_123p1_coordinates_tmp.js

 #python3 project_distance.py ../dataset_3D/123_dataset_new/250/data_mat_1_250.csv ../dataset_3D/123_dataset_new/250/data_mat_2_250.csv ../dataset_3D/123_dataset_new/250/data_mat_3_250.csv ../html3Dviz/123_123p2_coordinates_tmp.js
#python3 project_distance.py ../dataset_3D/123_dataset_new/250/data_mat_1_250.csv ../dataset_3D/123_dataset_new/250/data_mat_2_250.csv ../dataset_3D/123_dataset_new/250/data_mat_3_250.csv  ../html3DvizProj/proj123_123p2_coordinates_tmp.js
'''
d1path='../dataset_tabluar/dissimple1000_1.csv'
d2path='../dataset_tabluar/dissimple1000_2.csv'
d3path='../dataset_tabluar/dissimple1000_3.csv'
out_info_path='../html3Dviz/simple1000_123p1_coordinates_tmp.js'
'''
d1path=sys.argv[1]
d2path=sys.argv[2]
d3path=sys.argv[3]
out_info_path=sys.argv[4]
 
d1=pd.read_csv(d1path, header=None)
d2=pd.read_csv(d2path, header=None)
d3=pd.read_csv(d3path,header=None)

print("shape of ground truth data:", d1.shape)

outp1,outp2,outp3= processfile(out_info_path)
print("output 2D points:", len(outp1))


outd1=euclidean_distances(outp1)
outd2=euclidean_distances(outp2)
outd3=euclidean_distances(outp3)
 
a=find_distortion(d1,outd1)
b=find_distortion(d2,outd2)
c=find_distortion(d3,outd3) 
print("projection 1 Distortion:",int(a))
print("projection 2 Distortion:",int(b)) 
print("projection 3 Distortion:",int(c))
print("avg:",int((a+b+c)/3))
