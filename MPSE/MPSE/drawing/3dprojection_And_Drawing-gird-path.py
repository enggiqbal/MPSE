import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import sys
import pdb
import matplotlib.lines as mlines
import networkx as nx
def get2Dproject(points, P,W):
    Z=np.min( np.sum(P,axis=1))
    A= np.sum(P,axis=1) !=Z
    svd=np.linalg.svd(P)
    print("svd1:",svd[1])
    proj=svd[2]
    #print("projection",P)
    print("projection svd", proj)

    #proj=proj[:,0:2]
    proj=proj[0:2,:]
    proj = proj.transpose()
    #proj=proj[:,W]

#    print("projection--->", proj)
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
def draw2d(points,outdir,js_file_name,proj, type):
    filename=outdir+"edges_"+js_file_name.replace(".js", proj) + ".png"
    fig = plt.figure()
    ax = plt.axes()
    ax.grid(False)
    plt.axis('off')
    ax.scatter(points.T[0], points.T[1],  c='green', marker="x" ,  cmap='Greens');
    #pdb.set_trace()
    if type=="grid":
        n=int(np.sqrt(len(points)))
        G=nx.grid_graph(dim=[n,n])
        for i in range(0, n):
            for j in range(0,n):
                pos=points[n*i+j]
                G.add_node((i, j),pos=pos)
        for e in G.edges():
            i=G.node[e[0]]['pos']
            j=G.node[e[1]]['pos']
            line = mlines.Line2D([i[0],j[0]], [i[1],j[1]], color='green')
            ax.add_line(line)
    else:
        for i in range(0, len(points)-1):
            line = mlines.Line2D([points[i][0],points[i+1][0]], [points[i][1],points[i+1][1]], color='green')
            ax.add_line(line)


    plt.savefig(filename)




    #plt.savefig(filename)

proj=""
points=""
costhistory=""
def processfile(js_file_name):
    outdir="pngs_grid_path/"
    indir="../html3Dviz/"
    indir="../html3DvizProj/"
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

    P=np.array(proj)
    #print(P.shape)
    #print(P)
    points_txt=lines[11].replace("var","").replace(" ","")
    exec(points_txt, globals())
    points_data=np.array(points)
    #print(points_data.shape)

    costhistory_txt= lines[3].replace("var","").replace(" ","")
    exec(costhistory_txt, globals())
    costhistory_data=np.array(costhistory)



    ps1=get2Dproject(points_data, P[0],[True, True, False])
    ps2=get2Dproject(points_data, P[1],[False, True, True])
    ps3=get2Dproject(points_data, P[2],[True, False, True])


    draw2d(ps1,outdir,js_file_name,"projection1", "grid")
    draw2d(ps2,outdir,js_file_name,"projection2", "path")
    draw2d(ps3,outdir,js_file_name,"projection3","path")



    x=np.arange(len(costhistory))
    fig = plt.figure()
    ax = plt.axes()
    #ax.scatter(x.T, costhistory.T,  c='green', marker="x" ,  cmap='Greens');
    plt.plot(x,costhistory_data )
    plt.savefig(outdir+ "cost"+ js_file_name.replace(".js",".png"))







import pandas as pd

theader='''
     \\begin{table}[h!]
     \\begin{center}
     \\begin{tabular}{ | p{3cm} |c |c |c|c| }
     \\hline
'''

tfooter='''

      \\end{tabular}
      \\caption{ Analysis}
      \\label{tbl:pic}
      \\end{center}
      \\end{table}

'''

latextxt='''
@@@info@@@
&
 \\includegraphics[width=0.1\\textwidth]{figures/results2/@@@costpng@@@}
      &
     \\includegraphics[width=0.1\\textwidth]{figures/results2/@@@projection1@@@}
      &
              \\includegraphics[width=0.1\\textwidth]{figures/results2/@@@projection2@@@}

           &
               \\includegraphics[width=0.1\\textwidth]{figures/results2/@@@projection3@@@}
        \\\ \hline
'''
#latexcode.write(theader)
#for i in range(0, list.shape[0]):
job_name="projgrid_12p1"

js_file_name=job_name+ "_coordinates_tmp.js"
info=job_name.replace("_", "-")  #+ " PSet:"+ job_name[::-1][0] + " WSet:" +str(len(job_name.split("_")[1].split("p")[0]))
processfile(js_file_name)




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
