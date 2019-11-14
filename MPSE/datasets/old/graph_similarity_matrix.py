import networkx as nx
import numpy as np
import pygraphviz as pgv
"""
GetSimlarityMatrix takes a input graph (dot format) and produce outputs
similarity matrix,
networkx graph,
the node list which is a mapping between similarity matrix and graph nodes
"""
def get_similarity_matrix(dotpath):
    G=nx.MultiGraph()
    nodes_index={}
    G=nx.MultiGraph(pgv.AGraph(dotpath))
    SP=nx.all_pairs_shortest_path_length(G)
    n=len(G.nodes())
    count=0
    for node in G.nodes():
        nodes_index[node]=count
        count=count+1
    M = np.zeros(shape=(n,n))
    for x in SP:
        for y in x[1]:
            i=nodes_index[x[0]]
            j=nodes_index[y]
            M[i][j]=x[1][y]
    return M, G, nodes_index

#dotpath='../dataset/total_graph.dot'
#M,_,_=GetSimlarityMatrix(dotpath)
#print(M)
