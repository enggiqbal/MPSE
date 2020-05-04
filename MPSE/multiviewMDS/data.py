import numpy as np
import pandas as pd
import time, os
def get_matrix(csv):
    df=pd.read_csv(csv, header=None)
    M=df.values
    return M


def js_data_writer(A,file_path,costs,P): 
    localtime = time.asctime( time.localtime(time.time()) )
    pos_tmp=A
    jsdata ="var points="+ str(  pos_tmp.tolist()) + ";"
    print("Saving js data in: "+  file_path,flush=True)
    f=open(file_path,"w")
    f.write("var t='"+ localtime +"';\n")
    f.write("var steps={0};\n ".format(len(costs)))
    costhistory="\nvar costhistory="+ np.array2string(costs, precision=2, separator=',') + ";\n"
    proj="["
    for x in P:
        proj=proj + np.array2string(x, precision=6, separator=',', suppress_small=True) + ","
    proj=proj+"]"
    f.write("var proj="+proj + ";\n")
    f.write(jsdata)
    f.write(costhistory)
    f.close()
    print("JS file was saved in: ", file_path)
