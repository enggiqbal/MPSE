import numpy as np
import pandas as pd
import time, os
def get_matrix(csv):
    df=pd.read_csv(csv, header=None)
    M=df.values
    return M

#dotpath='../dataset/total_graph.dot'
#M,_,_=GetSimlarityMatrix(dotpath)
#print(M)


def js_data_writer(A,file_path,costs, P1, P2, P3):
    localtime = time.asctime( time.localtime(time.time()) )
    pos_tmp=A
    jsdata ="var points="+ str(  pos_tmp.tolist()) + ";"
    f=open(file_path,"w")
    f.write("var t='"+ localtime +"';\n")
    f.write("var steps={0};\n ".format(len(costs)))
    costhistory="var costhistory="+ str( costs) + ";\n"
    f.write(costhistory)
    #proj=proj+"[["+str(P1)+"],["+str(P2)+"],["+str(P3)+"]],"
    #proj="["+str(list(P1.T[0]))+","+str(list(P2.T[0]))+","+str(list(P3.T[0]))+"]"
    proj="["+np.array2string(P1, precision=6, separator=',', suppress_small=True)+","+np.array2string(P2, precision=6, separator=',', suppress_small=True)+","+np.array2string(P3, precision=6, separator=',', suppress_small=True)+"]"
    #pdb.set_trace()
    f.write("var proj="+proj + ";\n")
    f.write(jsdata)
    
    f.close()
    print("JS file was saved in: ", file_path)

def create_viz_file(outputpath,name_data_set,js_file_name):
    tem=open( os.path.join("resources",  "viz_distance_matrix_template.html"), 'r')
    html=tem.read()
    html=html.replace("###JS_FILE_NAME###",js_file_name)
    file_path=os.path.join( outputpath, name_data_set +"_viz.html")
    f=open(file_path,"w")
    f.write(html)
    f.close()
