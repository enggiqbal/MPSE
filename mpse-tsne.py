# python3.6 mpse.py -ds 123

import argparse
import math
import pandas as pd
import numpy as np
import os
import sys
import MPSE.mview as mview
import matplotlib.pyplot as plt
import shutil
import random
import time


def load123data(sample_size):
    path = 'MPSE/mview_examples/data/123/input/'
    Y1 = np.genfromtxt(path+'spicy_rice_10000_1.csv', delimiter=',')
    Y2 = np.genfromtxt(path+'spicy_rice_10000_2.csv', delimiter=',')
    Y3 = np.genfromtxt(path+'spicy_rice_10000_3.csv', delimiter=',')
    sample_size = min(len(Y1),  sample_size)
    sub = np.array(random.sample(range(len(Y1)), sample_size))
    return [Y1[sub], Y2[sub], Y3[sub]]


def load_data(args):
    D = [pd.read_csv(f, header=None).values for f in args.d]
    print(f"Total Samples found:%d<br>" % len(D[0]))
    args.sample_size = min(len(D[0]), args.sample_size)
    sub = range(args.sample_size)
    D = [(a[sub])[:, sub] for a in D]
    return D


def js_data_writer(A, file_path, costs, P):
    localtime = time.asctime(time.localtime(time.time()))
    pos_tmp = A
    jsdata = "var points=" + str(pos_tmp.tolist()) + ";"
    f = open(file_path, "w")
    f.write("var t='" + localtime + "';\n")
    f.write("var steps={0};\n ".format(len(costs)))
    costhistory = "\nvar costhistory=" + \
        np.array2string(costs, precision=2, separator=',') + ";\n"
    proj = "["
    for x in P:
        proj = proj + np.array2string(x, precision=6,
                                      separator=',', suppress_small=True) + ","
    proj = proj+"]"
    f.write("var proj="+proj + ";\n")
    f.write(jsdata)
    f.write(costhistory)
    f.close()
   


def get_matrix(args):
    if args.preloaded_dataset == '123':
        D = load123data(args.sample_size)

    # if args.preloaded_dataset == 'credit':
    #     path = 'MPSE/datasets/dataset_tabluar/data/'
    #     args.d = [path+'discredit3_tsne_1000_1.csv',  path +
    #               'discredit3_tsne_1000_2.csv', path+'discredit3_tsne_1000_3.csv']
    #     D = load_data(args)


    if args.preloaded_dataset == 'credit':
        path = 'MPSE/datasets/dataset_tabluar/data/'
        args.d = [path+'discredit3_tsne_cluster_1000_1.csv',  path +
                  'discredit3_tsne_cluster_1000_2.csv', path+'discredit3_tsne_cluster_1000_3.csv']
        D = load_data(args)

    if args.preloaded_dataset == 'circlesquare':
        path = 'MPSE/datasets/dataset_3D/circle_square_new/'
        args.d = [path+'dist_circle.csv', path+'dist_square.csv']
        D = load_data(args)
    if args.preloaded_dataset=='cluster':
        path = 'MPSE/datasets/dataset_3D/clusters_dataset/'
        args.d = [path+'dist_1.csv', path+'dist_2.csv', path+'dist_3.csv']
        D = load_data(args)
    if args.preloaded_dataset=='cupmouse':
        path = 'MPSE/datasets/dataset_3D/cup_mouse/'
        args.d = [path+'cup.csv', path+'mouse.csv' ]
        D = load_data(args)
    if args.preloaded_dataset == None:
        D = load_data(args)
    return D


def write_output(mv, args):
    projections = mv.Q
    # import pdb; pdb.set_trace()
    costs = np.array ([mv.cost])

    pos = mv.X
    args.output_dir = 'MPSE/outputs/' + args.experiment_name + "/"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # path_to_copy = "cp -rf MPSE/resources/vistemplateattributebased/* "
    # if args.visualization_template == "pointbased":
    path_to_copy = "cp -rf MPSE/resources/vistemplatepointbased/* "
    os.system(path_to_copy + args.output_dir)
    f = open(args.output_dir+"/vis_param.js", "r")
    vis_params = f.read().replace("var numberofprojection=3;",
                                 "var numberofprojection=" + str(len(projections)) + ";")
    vis_params=vis_params+" var backgroundcolor='"+  str(args.bgcolor) + "';\n " 
    vis_params=vis_params+" var pointcolor='"+  str(args.pcolor) + "';\n " 

    f.close()
    f = open(args.output_dir+"/vis_param.js", "w+")
    f.write(vis_params)
    f.close()
    js_file_path = os.path.join(args.output_dir, "coordinates.js")
    js_data_writer(pos, js_file_path, costs, projections)

    posfile = os.path.join(args.output_dir, args.experiment_name + "_pos.csv")
    np.savetxt(posfile, pos, delimiter=",")
    costfile = os.path.join(
        args.output_dir, args.experiment_name + "_costs.csv")
    np.savetxt(costfile, costs, delimiter=",")

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(np.arange(len(costs)), costs)
    costfile = os.path.join(args.output_dir, "cost.png")
    plt.savefig(costfile)
    sys.stdout.flush()
    print("<br> output path: " + os.path.join(args.output_dir) + "index.html")
    print("<br><h1> <a target='_blank'  href ='static/" + args.experiment_name +
          "/index.html'>Interactive visualization</a></h1><br>", flush=True)
    print("<br><h2> <a target='_blank'  href ='static/" + args.experiment_name + "/" + args.experiment_name + "_pos.csv"
          "'>Download 3D positions</a></h2><br>", flush=True)
    print("<br><h2> <a target='_blank'  href ='static/" + args.experiment_name +
          "/coordinates.js'>Output details (history, projections, position) was saved here</a></h2><br>", flush=True)
 
    print("<br><img src=/static/"+args.experiment_name + "/cost.png" + ">")


if __name__ == '__main__':
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
    parser.add_argument('-X0', '--X0', default='False', 
                        choices=['True', 'False'], help='Smart initialization', required=False)
    parser.add_argument('-ps', '--projection_type',  default='standard', choices=['fixed',
                                                                                  'same', 'standard', 'cylinder', 'orthogonal', 'normal', 'uniform', 'variable'], help="projection set", required=False)
    parser.add_argument('-vt', '--visualization_template',  default='pointbased', choices=[
                        'pointbased', 'attributebased'], help="Visualization template", required=False)
    parser.add_argument('-an', '--average_neighbors', type=int,
                        default=32, help="average  neighbors", required=False)
    parser.add_argument('-ds', '--preloaded_dataset', default=None,
                        help="Preloaded Dataset", required=False)

    parser.add_argument('-bgcolor', '--bgcolor', default='black',
                            help="vis: background color", required=False)
    parser.add_argument('-pcolor', '--pcolor', default='red',
                            help="vis: poing color", required=False)
                            
    args = parser.parse_args()
    if args.preloaded_dataset == None and args.d == None:
        print("Please provide either -d ( distace matices) or -ds (preloaded dataset)")
        exit(1)

    print("<h1>Please keep the window running</h1>")

    D = get_matrix(args)

    args.projection_type = None if args.projection_type == 'variable' else args.projection_type
    # import pdb; pdb.set_trace()

    if args.X0=="True":
        args.X0=True
    else:
        args.X0=False
    
    # mv = mview.basic(D,visualization_method = 'tsne', Q=args.projection_type, verbose=2, smart_initialize=args.X0,
    #                  max_iter=args.max_iters, average_neighbors=args.average_neighbors, min_cost=args.min_cost)

    
    mv = mview.basic(D, verbose=2, visualization_args = {'perplexity':100}, max_iters=200,  visualization_method = 'tsne')
    # mv = mview.basic(D, verbose=2, max_iters=200)
                     
    mv.plot_computations()
    # mv.plot_images()
    # write_output(mv, args)
