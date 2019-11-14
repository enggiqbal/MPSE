import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import pdb
import time;

 
from mview_call import standard
from mview_call import example_standard
from mview_call import example_main
def temp_data_writer(A,file_path,costs, P1, P2, P3, i, newcost):
    localtime = time.asctime( time.localtime(time.time()) )
    #pos_tmp=A.reshape(int(len(A)/3),3)
    pos_tmp=A
    jsdata ="var points="+ str(  pos_tmp.tolist()) + ";"
    f=open(file_path,"w")
    f.write("var t='"+ localtime +"';\n")
    f.write("var steps={0};\n  var cost={1};\n".format(i,newcost))
    #pdb.set_trace()
    costhistory="var costhistory="+ str(costs.tolist()) + ";\n"
    f.write(costhistory)
    proj="["+np.array2string(P1, precision=6, separator=',', suppress_small=True)+","+np.array2string(P2, precision=6, separator=',', suppress_small=True)+","+np.array2string(P3, precision=6, separator=',', suppress_small=True)+"]"
    f.write("var proj="+proj + ";\n")
    f.write(jsdata)
    f.close()
 




def create_viz_file(outputpath,name_data_set,js_file_name):
    tem=open(outputpath+ "viz_credit_template.html", 'r')
    html=tem.read()
    html=html.replace("###JS_FILE_NAME###",js_file_name)
    file_path=outputpath+name_data_set +"_viz.html"
    f=open(file_path,"w")
    f.write(html)
    f.close()

#python3.6 mview_hpc.py credit4fix500_3 500 3 1 html3Dviz/
#python3.6 mview_hpc.py credit4fix500_2 500 2 1 html3Dviz/

#python3.6 mview_hpc.py credit4var500_3 100 3 0 html3Dviz/
 


name_data_set=sys.argv[1]
number_of_points=int(sys.argv[2])
projection_set=int(sys.argv[3])
fixed=int(sys.argv[4])
outputpath='MPSE/credit/'


js_file_path=name_data_set +"_coordinates_tmp.js"
if fixed==1:
    points,proj,cost,costhistory=example_standard(number_of_points=number_of_points, number_of_projs=projection_set)
  
    proj=np.array([[[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.]],
       [[1., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],
       [[0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]])
else:
    points,proj,cost,costhistory=example_main(number_of_points=number_of_points, number_of_projs=3)
 
#creating viz file from template
create_viz_file(outputpath,name_data_set,js_file_path)
#save data
temp_data_writer(points,outputpath+js_file_path,costhistory, proj[0], proj[1], proj[2], 0, cost)

#python3.6 mview_hpc.py credit4 dataset_tabluar/data/discredit3_1000_1.csv dataset_tabluar/data/discredit3_1000_2.csv dataset_tabluar/data/discredit3_1000_3.csv  0.001 10000 html3Dviz/ 1 3 
