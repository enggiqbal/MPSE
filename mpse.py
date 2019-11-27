#python3 mpse.py -d datasets/dataset_tabluar/data/dissimple1000_1.csv  datasets/dataset_tabluar/data/dissimple1000_2.csv -d3 datasets/dataset_tabluar/data/dissimple1000_3.csv

import argparse
import math
from MPSE.multiviewMDS import multiview_core 
from MPSE.multiviewMDS import data
import numpy as np
import os, sys
import MPSE.mview as mview
import matplotlib.pyplot as plt
import shutil

parser = argparse.ArgumentParser(description='MPSE')
parser.add_argument('-d', '--d', type=argparse.FileType('r'), nargs='+', help='List of input files with distace matices', required=True)
parser.add_argument('-o','--output_dir', default='outputs', help='Output directory',required=False)
parser.add_argument('-e','--experiment_name', default='exp', help='Experiment name',required=False)

parser.add_argument( '-p','--projections', type=int,default=3,   help='Number of projections to optimize',required=False)
parser.add_argument( '-t','--projections_type', default='fixed', choices=["fixed","variable"], help='Projection type',required=False)
#parser.add_argument( '-ps','--projection_set', default='resources/fixed_projection_1.txt', help='file for projection set for fixed projection, see examples in resource directory',required=False)


parser.add_argument( '-lr','--lr', type=float,default=0.0001, help='Learning rate',required=False)
parser.add_argument( '-max_iters','--max_iters', type=int,default=10000, help='Max iterations',required=False)
parser.add_argument( '-n','--sample_size',type=int, default=math.inf , help='Number of samples',required=False)
parser.add_argument( '-X0','--X0', default=None, help='Initial initialization, a csv file with 3D co-ordinates',required=False)
parser.add_argument( '-sp','--save_progress',type=int,  default=0, help='save progress',required=False)
parser.add_argument( '-v','--verbose',type=int,  default=2, help='verbose',required=False)
parser.add_argument( '-alg','--algorithm',  default='MULTIVIEW', choices=['classic','gd','gdm','agd','MULTIVIEW0','MULTIVIEW'], help="algorithms: 'classic' for autograd implementation,\n  'gd' for gradient descent,\n 'gdm' for GD with momentum, \n 'agd' for adaptive GD",required=False)
parser.add_argument( '-ps','--projection_set',  default='standard', choices=[ 'same', 'standard', 'cylinder', 'orthogonal', 'normal', 'uniform'], help="projection set",required=False)
parser.add_argument( '-vt','--visualization_template',  default='pointbased', choices=[ 'pointbased', 'attributebased'], help="visualization template",required=False)



args = parser.parse_args()
print("<h1>Please keep the window running</h1>")
if (args.verbose):
    print("================= list of parameters ================= ")
    print(args)
    sys.stdout.flush()

#assert len(args.d)>2 , "At least 3 inputs distance matrix is required."
D=[data.get_matrix(f) for f in args.d]

if (args.verbose):
    print("Total Samples: ", len(D[0]))

args.sample_size=min(len(D[0]), args.sample_size)
sub = range(args.sample_size)
D=[(a[sub])[:,sub] for a in D ]


#if args.projections_type=='fixed' and args.algorithm=='classic':
pfile="MPSE/resources/fixed_projection_1.txt"
if  args.projection_set=='cylinder':
    pfile="MPSE/resources/fixed_projection_2.txt"
txt = open(pfile,"r+").read()
txt=txt.replace("\n","")
P=eval(txt)

if args.projections >3:
    assert len(P)==args.projections, "Provide correct  projection sets and projection number"

if args.X0 : 
    A=data.get_matrix( args.X0)
    assert args.sample_size==len(A), "initialization should matched with sample size"
else:
    A=np.random.rand(args.sample_size*3,1)


 
eps=1e-9
stopping_eps=0.1
'''
if args.algorithm=='classic':
    mview=multiview_core.multiview(D, P, 3, eps, args.projection_set,args.projections)
    pos,costs, projections=mview.multiview_mds(A,args.max_iters, args.lr, stopping_eps,args.output_dir,args.experiment_name, args.save_progress, args.verbose)
    pos=pos.reshape(int(len(A)/3),3)
'''
if args.projections_type=='fixed':
    D = D[0:args.projections]
    pos,_,costs=mview.MULTIVIEW0(D,Q=args.projection_set,X0=None, lr=args.lr,max_iters=args.max_iters,verbose=args.verbose)
    projections=P
else:
    D = D[0:args.projections]
    pos,projections,_,costs=mview.MULTIVIEW(D,X0=None,max_iters=args.max_iters,verbose=args.verbose)

#print(projections)
#write to file
args.output_dir='MPSE/outputs/'+ args.experiment_name + "/"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if args.visualization_template=="pointbased":
    os.system("cp -rf MPSE/resources/vistemplatepointbased/* " + args.output_dir)
else:
    os.system("cp -rf MPSE/resources/vistemplateattributebased/* " + args.output_dir)

f=open(args.output_dir+"/vis_param.js","r")
vis_param=f.read().replace("var numberofprojection=3;","var numberofprojection="+ str(args.projections ) +";")
f.close()
f=open(args.output_dir+"/vis_param.js","w+")
f.write(vis_param)
f.close()

js_file_path=os.path.join(args.output_dir, "coordinates.js")
data.js_data_writer(pos,js_file_path,costs, projections)

posfile=os.path.join(args.output_dir, args.experiment_name +"_pos.csv")
np.savetxt(posfile, pos, delimiter=",")
costfile=os.path.join(args.output_dir, args.experiment_name +"_costs.csv")
np.savetxt(costfile, costs, delimiter=",")
 






x=np.arange(len(costs))
fig = plt.figure()
ax = plt.axes()
plt.plot(x,costs )
costfile=os.path.join( args.output_dir, "cost.png")
plt.savefig(costfile)
print("cost history saved as ",costfile )
print("<img src=/static/"+args.experiment_name +  "/cost.png" +">" )
sys.stdout.flush()

print ("<br><h1> <a target='_blank'  href ='static/"+ args.experiment_name +"/index.html'>interactive visualization</a></h1><br>", flush=True)
 

if (args.verbose):
    print("Cost history was saved in: ", costfile)
    print("Output 3D position was saved in: ", posfile)




#TODO: to be continued python3 mpse.py -d datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv -p 2 -lr 0.001 -n 10 -alg classic
