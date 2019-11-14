import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import pdb
import time;
from gmds import mds, special
import gmds_call



# Vahan's comment





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
    tem=open(outputpath+ "viz_distance_matrix_template.html", 'r')
    html=tem.read()
    html=html.replace("###JS_FILE_NAME###",js_file_name)
    file_path=outputpath+name_data_set +"_viz.html"
    f=open(file_path,"w")
    f.write(html)
    f.close()




name_data_set=sys.argv[1]
dpath1=sys.argv[2]
dpath2=sys.argv[3]
dpath3=sys.argv[4]

alpha= float(sys.argv[5])
steps=int(sys.argv[6])
outputpath=sys.argv[7]
projection_set=int(sys.argv[8])
number_of_weights=int(sys.argv[9])

D1 = np.genfromtxt(dpath1, delimiter=',')
D2 = np.genfromtxt(dpath2, delimiter=',')
D3 = np.genfromtxt(dpath3, delimiter=',')

js_file_path=name_data_set +"_coordinates_tmp.js"

#call main function
#points, proj, cost, costhistory, Qs, X0, proj0, Q0s=main(D1,D2,D3,False)
points,proj,cost,costhistory,Qs,X0,proj0,Q0s = gmds_call.main(D1,D2,D3,     feedback=True)



#entry the job in index.html file
html="name_data_set:" + str(name_data_set) + " dpath1:" + str(dpath1)+ " alpha:" + str(alpha) + " steps:" + str(steps)+ " outputpath:" + str(outputpath)  + " projection_set:" + str(projection_set) + " number_of_weights:" + str(number_of_weights)
html = "<a href="+name_data_set +"_viz.html" +">"+ html+"</a><br>"
f=open(outputpath+"index.html","a")
f.write(html)
f.close()

#creating viz file from template
create_viz_file(outputpath,name_data_set,js_file_path)
#save data
temp_data_writer(points,outputpath+js_file_path,costhistory, proj[0], proj[1], proj[2], 0, cost)
