import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys
import pdb


def processor(list,exclude,html3Dviz,path):

    theader='<table width=70%><tr><td>Data and projection set</td><td>1st projection </td><td>2nd projection</td><td>3rd projection</td><td>ulr</td></tr>'
    tfooter='</table>'
    tbody=""
    for i in range(0, len(list)):
        #import pdb; pdb.set_trace()
        #print(list[i])
        job_name=list[i].strip()
        if job_name in exclude: continue
        js_file_name=job_name+ "_coordinates_tmp.js"
        info=job_name.split("_")[0] +   " PSet:"+ job_name[::-1][0] #+ " WSet:" +str(len(job_name.split("_")[1].split("p")[0]))
        costfile="cost"+ js_file_name.replace(".js",".png")
        projection1=js_file_name.replace(".js", "projection1") + ".png"
        projection2=js_file_name.replace(".js", "projection2") + ".png"
        projection3=js_file_name.replace(".js", "projection3") + ".png"
        prefix=""
        tmppath=path
        if "cluster" in job_name:
            tmppath="drawing/pngs_cluster/"
            prefix="cluster_"
        if "grid" in job_name:
            tmppath="drawing/pngs_grid_path/"
            prefix="edges_"
        if "cir" in  job_name:
            tmppath="drawing/pngs_k_path/"
            prefix="edges_"

        url=html3Dviz+ job_name + "_viz.html"
        #path="drawing/pngs/"
        td0="<td>"+ info + "</td>"
        td1="<td><img width=50% src='" +tmppath+prefix+projection1+"'/></td>"
        td2="<td><img width=50% src='"+tmppath+prefix+projection2+"'/></td>"
        td3="<td><img width=50% src='"+tmppath+prefix+projection3+"'/></td>"
        td4="<td><a target='_blank' href='"+url+"'/>interaction in 3D space</a></td>"
        if len(job_name.split("_")[1].split("p")[0])==2:
            td3="<td></td>"
        tbody=tbody+'<tr>' +td0+td1 + td2+td3 + td4+'</tr>\n'
    return theader + tbody + tfooter




import pandas as pd

code=open("../index.html","w")

list=pd.read_csv( '../hpc/joblist.csv', header=None)
code.write('''
<html><head>
<style>
body{ margin:0 auto; }
 table { width: 50%; margin-left: auto; margin-right: auto; }
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
</style>
<title>3D Multiview Graph Visualization (3DMGV) </title></head><body align=center>
<h1>Same Graph, Different Views  (3D Multiview Graph Visualization)<h1><br>
<h1> Experiment Results</h1>

<h2>Fixed Projection</h2>
''')
list=list[0].values
exclude=["projcupmouse_123p1","projconfname_123p1","confname_123p1","cluster_12p2"]
html=processor(list,exclude,"html3Dviz/", "drawing/pngs/")

code.write(html)
code.write("<h2> Variable Projection<h2>")
list2=pd.read_csv( '../hpc/proj_joblist.csv', header=None)
list2=list2[0].values
html=processor(list2,exclude,"html3DvizProj/", "drawing/proj_pngs/")
code.write(html)
code.write("<body></html>")
code.close()
