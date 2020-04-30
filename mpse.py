# python3 mpse.py -d datasets/dataset_tabluar/data/dissimple1000_1.csv  datasets/dataset_tabluar/data/dissimple1000_2.csv -d3 datasets/dataset_tabluar/data/dissimple1000_3.csv

import argparse
import math
from MPSE.multiviewMDS import multiview_core
from MPSE.multiviewMDS import data
import numpy as np
import os
import sys
import MPSE.mview as mview
import matplotlib.pyplot as plt
import shutil

parser = argparse.ArgumentParser(description='MPSE')
parser.add_argument('-d', '--d', type=argparse.FileType('r'), nargs='+',
                    help='List of input files with distace matices', required=True)
parser.add_argument('-o', '--output_dir', default='outputs',
                    help='Output directory', required=False)
parser.add_argument('-e', '--experiment_name', default='exp',
                    help='Experiment name', required=False)

#parser.add_argument( '-p','--projections', type=int,default=3,   help='Number of projections to optimize',required=False)
#parser.add_argument( '-t','--projection_type', default='fixed', help='Projection type',required=False)
#parser.add_argument( '-ps','--projection_set', default='resources/fixed_projection_1.txt', help='file for projection set for fixed projection, see examples in resource directory',required=False)

#parser.add_argument( '-lr','--lr', type=float,default=0.0001, help='Learning rate',required=False)
parser.add_argument('-max_iters', '--max_iters', type=int,
                    default=10000, help='Max iterations', required=False)
parser.add_argument('-n', '--sample_size', type=int,
                    default=math.inf, help='Number of samples', required=False)
parser.add_argument( '-X0','--X0', default=None, help='Initial initialization',required=False)
#parser.add_argument( '-sp','--save_progress',type=int,  default=0, help='save progress',required=False)
#parser.add_argument( '-v','--verbose',type=int,  default=2, help='verbose',required=False)
#parser.add_argument( '-alg','--algorithm',  default='MULTIVIEW', choices=['classic','gd','gdm','agd','MULTIVIEW0','MULTIVIEW'], help="algorithms: 'classic' for autograd implementation,\n  'gd' for gradient descent,\n 'gdm' for GD with momentum, \n 'agd' for adaptive GD",required=False)
parser.add_argument('-ps', '--projection_type',  default='standard', choices=['fixed',
                    'same', 'standard', 'cylinder', 'orthogonal', 'normal', 'uniform', 'variable'], help="projection set", required=False)
parser.add_argument('-vt', '--visualization_template',  default='pointbased', choices=[
                    'pointbased', 'attributebased'], help="visualization template", required=False)
parser.add_argument('-an', '--average_neighbors', type=int,
                    default=32, help="average  neighbors", required=False)

args = parser.parse_args()
print("<h1>Please keep the window running</h1>")
D = [data.get_matrix(f) for f in args.d]
print(f"Total Samples found:%d<br>" % len(D[0]))

args.sample_size = min(len(D[0]), args.sample_size)
sub = range(args.sample_size)
D = [(a[sub])[:, sub] for a in D]


if args.projection_type == 'variable':
    args.projection_type = None
if args.X0=='mds':
    args.X0=True
else:
    args.X0=False
    
#print("args.projection_type="+args.projection_type)
#print(f"max_iters=%d" % (args.max_iters))
mv = mview.basic(D,  Q=args.projection_type, verbose=2, smart_initialize=args.X0, max_iter=args.max_iters, average_neighbors=args.average_neighbors)

projections = mv.Q
costs = mv.H["costs"]
pos = mv.X

# write to file
args.output_dir = 'MPSE/outputs/' + args.experiment_name + "/"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if args.visualization_template == "pointbased":
    os.system("cp -rf MPSE/resources/vistemplatepointbased/* " + args.output_dir)
else:
    os.system("cp -rf MPSE/resources/vistemplateattributebased/* " + args.output_dir)

f = open(args.output_dir+"/vis_param.js", "r")
vis_param = f.read().replace("var numberofprojection=3;",
                             "var numberofprojection=" + str(len(projections)) + ";")
f.close()
f = open(args.output_dir+"/vis_param.js", "w+")
f.write(vis_param)
f.close()

js_file_path = os.path.join(args.output_dir, "coordinates.js")
data.js_data_writer(pos, js_file_path, costs, projections)

posfile = os.path.join(args.output_dir, args.experiment_name + "_pos.csv")
np.savetxt(posfile, pos, delimiter=",")
costfile = os.path.join(args.output_dir, args.experiment_name + "_costs.csv")
np.savetxt(costfile, costs, delimiter=",")


x = np.arange(len(costs))
fig = plt.figure()
ax = plt.axes()
plt.plot(x, costs)
costfile = os.path.join(args.output_dir, "cost.png")
plt.savefig(costfile)
sys.stdout.flush()
print("<br><h1> <a target='_blank'  href ='static/" + args.experiment_name +
      "/index.html'>interactive visualization</a></h1><br>", flush=True) 
print("Output 3D position was saved in: ", posfile)
print("cost history saved as ", costfile)
print("<img src=/static/"+args.experiment_name + "/cost.png" + ">")
# TODO: to be continued python3 mpse.py -d datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv -p 2 -lr 0.001 -n 10 -alg classic
