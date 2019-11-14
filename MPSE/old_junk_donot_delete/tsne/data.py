import networkx as nx
import numpy as np
import pygraphviz as pgv
import pandas as pd
def get_matrix(csv):
    df=pd.read_csv(csv, header=None)
    M=df.values
    return M

#dotpath='../dataset/total_graph.dot'
#M,_,_=GetSimlarityMatrix(dotpath)
#print(M)
