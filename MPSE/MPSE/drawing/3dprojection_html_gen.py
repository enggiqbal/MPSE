import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys
import pdb

def get_view_point(P):
    svd=np.linalg.svd(P)
    proj=svd[2]
    view_point=proj[2,:]
    return view_point
def create_viz_file(outputpath,name_data_set,js_file_name, viewpoints):
    tem=open("../html3Dviz/"+ "viz_distance_matrix_template.html", 'r')
    html=tem.read()
    html=html.replace("###JS_FILE_NAME###",js_file_name)
    html=html.replace("###VIEW_POINTS###",viewpoints)
    file_path=outputpath+name_data_set +"_viz.html"
    f=open(file_path,"w")
    f.write(html)
    f.close()



proj=""
points=""
costhistory=""
def processfile(job_name,js_file_name,indir):
    outdir="pngs/"


    global proj
    f=open(indir+js_file_name,"r")
    data=f.read()
    lines=data.split("\n")

    proj_txt=lines[4]+lines[5]+lines[6]+lines[7]+lines[8]+lines[9]+lines[10]
    proj_txt=proj_txt.replace("var","")
    proj_txt=proj_txt.replace("\n","")
    proj_txt=proj_txt.replace(";","")
    proj_txt=proj_txt.replace(" ","")
    #pdb.set_trace()
    exec(proj_txt, globals())

    P=np.array(proj)
    viewpoint1=np.array2string(get_view_point(P[0]),precision=2, separator=',')
    viewpoint2=np.array2string(get_view_point(P[1]),precision=2, separator=',')
    viewpoint3=np.array2string(get_view_point(P[2]),precision=2, separator=',')
    viewpoints='<script type="text/javascript">var zoomfactor=100; var view=Array(); view[0]='+str(viewpoint1)+'; view[1]='+str(viewpoint2)+'; view[2]='+str(viewpoint3)+';</script>'
    #js_file_path=name_data_set +"_coordinates_tmp.js"
    #self.create_viz_file(outputpath,name_data_set,js_file_path)
    #import pdb; pdb.set_trace()
    create_viz_file(indir,job_name,js_file_name, viewpoints)






import pandas as pd

list=pd.read_csv( '../hpc/proj_joblist.csv', header=None)
exclude=["confname_123p1","cluster_12p2"]
indir="../html3DvizProj/"
for i in range(0, list.shape[0]):
    job_name=list[0][i].strip()
    if job_name in exclude: continue
    js_file_name=job_name+ "_coordinates_tmp.js"
    info=job_name.replace("_", "-")  #+ " PSet:"+ job_name[::-1][0] + " WSet:" +str(len(job_name.split("_")[1].split("p")[0]))
    processfile(job_name,js_file_name,indir)



list=pd.read_csv( '../hpc/joblist.csv', header=None)
indir="../html3Dviz/"
for i in range(0, list.shape[0]):
    job_name=list[0][i].strip()
    if job_name in exclude: continue
    js_file_name=job_name+ "_coordinates_tmp.js"
    info=job_name.replace("_", "-")  #+ " PSet:"+ job_name[::-1][0] + " WSet:" +str(len(job_name.split("_")[1].split("p")[0]))
    processfile(job_name,js_file_name,indir)
