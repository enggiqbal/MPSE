# python3.6 mpse.py -ds 123

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
import random
parser = argparse.ArgumentParser(description='MPSE')
parser.add_argument('-d', '--d', type=argparse.FileType('r'), nargs='+',
                    help='List of input files with distace matices', required=False)
parser.add_argument('-o', '--output_dir', default='outputs',
                    help='Output directory', required=False)
parser.add_argument('-e', '--experiment_name', default='exp',
                    help='Experiment name', required=False)
parser.add_argument('-max_iters', '--max_iters', type=int,
                    default=1000, help='Max iterations', required=False)
parser.add_argument('-esp', '--min_cost', type=int,
                    default=0.001, help='stopping cost', required=False)

parser.add_argument('-n', '--sample_size', type=int,
                    default=math.inf, help='Number of samples', required=False)
parser.add_argument('-X0', '--X0', default=False, type=bool,
                    choices=[True, False], help='Smart initialization', required=False)
parser.add_argument('-ps', '--projection_type',  default='standard', choices=['fixed',
                                                                              'same', 'standard', 'cylinder', 'orthogonal', 'normal', 'uniform', 'variable'], help="projection set", required=False)
parser.add_argument('-vt', '--visualization_template',  default='pointbased', choices=[
                    'pointbased', 'attributebased'], help="Visualization template", required=False)
parser.add_argument('-an', '--average_neighbors', type=int,
                    default=32, help="average  neighbors", required=False)
parser.add_argument('-ds', '--preloaded_dataset', default=None,
                    help="Preloaded Dataset", required=False)


args = parser.parse_args()

if args.preloaded_dataset == None and args.d == None:
    print("Please provide either -d ( distace matices) or -ds (preloaded dataset)")
    exit(1)


def load123data(sample_size):
    path = 'MPSE/mview_examples/data/123/input/'
    Y1 = np.genfromtxt(path+'spicy_rice_10000_1.csv', delimiter=',')
    Y2 = np.genfromtxt(path+'spicy_rice_10000_2.csv', delimiter=',')
    Y3 = np.genfromtxt(path+'spicy_rice_10000_3.csv', delimiter=',')
    sample_size = min(len(Y1),  sample_size)
    sub = np.array(random.sample(range(len(Y1)), sample_size))
    return [Y1[sub], Y2[sub], Y3[sub]]


print("<h1>Please keep the window running</h1>")


def load_data(args):
    D = [data.get_matrix(f) for f in args.d]
    print(f"Total Samples found:%d<br>" % len(D[0]))
    args.sample_size = min(len(D[0]), args.sample_size)
    sub = range(args.sample_size)
    D = [(a[sub])[:, sub] for a in D]
    return D


if args.preloaded_dataset == '123':
    D = load123data(args.sample_size)

if args.preloaded_dataset == 'credit':
    args.d = ['MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv',
              'MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv', 'MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv']
    D = load_data(args)

if args.preloaded_dataset == 'circlesquire':
    args.d = ['MPSE/datasets/dataset_3D/circle_square_new/dist_circle.csv',
              'MPSE/datasets/dataset_3D/circle_square_new/dist_square.csv']
    D = load_data(args)

if args.preloaded_dataset == None:
    D = load_data(args)


if args.projection_type == 'variable':
    args.projection_type = None


mv = mview.basic(D,  Q=args.projection_type, verbose=2, smart_initialize=args.X0,
                 max_iter=args.max_iters, average_neighbors=args.average_neighbors, min_cost=args.min_cost)

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


#posfile = os.path.join(args.output_dir, args.experiment_name + "_projections.csv")
#np.savetxt(posfile, projections, delimiter=",")


x = np.arange(len(costs))
fig = plt.figure()
ax = plt.axes()
plt.plot(x, costs)
costfile = os.path.join(args.output_dir, "cost.png")
plt.savefig(costfile)
sys.stdout.flush()
print("<br> output path: "+ os.path.join(args.output_dir) + "index.html")

print("<br><h1> <a target='_blank'  href ='static/" + args.experiment_name +
      "/index.html'>Interactive visualization</a></h1><br>", flush=True)
#print("<br>Output 3D position was saved in: ", posfile)

print("<br><h2> <a target='_blank'  href ='static/" + args.experiment_name + "/" + args.experiment_name + "_pos.csv"
      "'>Output 3D position was saved here</a></h2><br>", flush=True)

print("<br><h2> <a target='_blank'  href ='static/" + args.experiment_name +
      "/coordinates.js'>Output details (history, projections, position) was saved here</a></h2><br>", flush=True)


print("<br>cost history saved as ", costfile)
print("<br><img src=/static/"+args.experiment_name + "/cost.png" + ">")
# TODO: to be continued python3 mpse.py -d datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv datasets/dataset_3D/circle_square_new/dist_circle.csv -p 2 -lr 0.001 -n 10 -alg classic
