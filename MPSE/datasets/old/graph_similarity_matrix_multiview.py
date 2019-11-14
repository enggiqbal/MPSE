import networkx as nx
import numpy as np
import pygraphviz as pgv
import pdb
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot
import os
import json
from networkx.readwrite import json_graph

"""
GetSimlarityMatrix takes a input graph (dot format) and produce outputs
similarity matrix,
networkx graph,
the node list which is a mapping between similarity matrix and graph nodes
"""
def connect_graph(G,c):
    #pdb.set_trace()
    print("",c, " comp" ,nx.number_connected_components(G))
    C=nx.connected_components(G)
    p=[]
    print("old:",len(G.edges()))
    for x in C:
        p.append(list(x)[0])
    G.add_path(p, color=c)
    print("adding ", c, " vertices:", len(p))

    #nx.add_path(G, p)
    print("new:",len(G.edges()))
    return G,p

def stat(dotpath):
    G_temp=nx.MultiGraph(pgv.AGraph(dotpath))
    c=0
    m=0
    s=0
    for node in G_temp.nodes(data=True):
        if node[1]['type3']=='Computer Science':
            c=c+1
        if node[1]['type3']=='Mathematics':
            m=m+1
        if node[1]['type3']=='Systems and Industrial Engr':
            s=s+1
    t=s*(s-1)/2 + m*(m-1)/2 + c*(c-1)/2
    e=0
    edgedata=nx.get_edge_attributes(G_temp,'edgetype')
    G=nx.Graph()
    G.add_nodes_from(G_temp.nodes(data=True))
    for x in edgedata:
        if edgedata[x]=='type3':
            e=e + 1
            G.add_edge(x[0],x[1])






def get_procssed_graph(dotpath):
    G=nx.MultiGraph()
    G_temp=nx.MultiGraph(pgv.AGraph(dotpath))
    nodes={}
    i =0
    for node in G_temp.nodes(data=True):
        nodes[node[0]]=i
        G.add_node(str(i),label=node[1]['label'])
        i=i+1
    print(nx.info(G))
    #pdb.set_trace()
    edgedata=nx.get_edge_attributes(G_temp,'edgetype')
    for e in edgedata:
        G.add_edge(str(nodes[e[0]]), str(nodes[e[1]]),edgetype=edgedata[e])

    #pdb.set_trace()
    G1=nx.Graph()
    G2=nx.Graph()
    G3=nx.Graph()
    G1.add_nodes_from(G.nodes(data=True))
    G2.add_nodes_from(G.nodes(data=True))
    G3.add_nodes_from(G.nodes(data=True))
    print("xxxxx")
    print(nx.info(G1))
    print(nx.info(G2))
    print(nx.info(G3))
    print(nx.info(G))
    print("xxxxx")
    for x in G.edges.data('edgetype'):
        if x[2]=='type1':#'dept':
            G1.add_edge(x[0],x[1], edgetype='type1', color='red')
        if  x[2]=='type2':
            G2.add_edge(x[0],x[1], edgetype='type2' ,color='green')
        if  x[2]=='type3':
            G3.add_edge(x[0],x[1], edgetype='type3' ,color='blue')
    print(nx.info(G1))
    print(nx.info(G2))
    print(nx.info(G3))
    print(nx.info(G))
    G1,p1=connect_graph(G1, 'red')
    G2,p2=connect_graph(G2,'green')
    G3,p3=connect_graph(G3,'blue')
    G.add_path(p1, color='red', edgetype='type1')
    G.add_path(p2, color='green',edgetype='type2')
    G.add_path(p3, color='blue',edgetype='type3')

    print(nx.info(G1))
    print(nx.info(G2))
    print(nx.info(G3))
    print(nx.info(G))



    write_dot(G1,'red.dot')
    write_dot(G2,'green.dot')
    write_dot(G3,'blue.dot')

    os.system(" sfdp -Goverlap=prism  -n  -Npenwidth=1  red.dot -Tpng > red.png")
    os.system(" sfdp -Goverlap=prism  -n  -Npenwidth=1  green.dot -Tpng > green.png")
    os.system(" sfdp -Goverlap=prism  -n  -Npenwidth=1  blue.dot -Tpng > blue.png")

    return G, G1, G2, G3, nodes

#,edgetype=edgetype[(e)]

def get_similarity_matrix(dotpath):
    G=nx.MultiGraph()
    nodes_index={}
    count=0
    G, G1, G2, G3,_=get_procssed_graph(dotpath)
    for node in G.nodes():
        nodes_index[node]=count
        count=count+1

    D1=get_matrix_from_SP(G1,nodes_index)
    D2=get_matrix_from_SP(G2,nodes_index)
    D3=get_matrix_from_SP(G3,nodes_index)
    return D1,D2,D3, G, nodes_index

def get_matrix_from_SP(G,nodes_index):
    SP=nx.all_pairs_shortest_path_length(G)
    n=len(G.nodes())
    D1 = np.zeros(shape=(n,n))
    for x in SP:
        for y in x[1]:
            i=nodes_index[x[0]]
            j=nodes_index[y]
            D1[i][j]=x[1][y]
    return D1
#dotpath='../dataset/total_graph.dot'
#M,_,_=GetSimlarityMatrix(dotpath)
#print(M)
#dotpath='total_graph.dot'
#dotpath='game_of_thrones_consistent.dot'
#D1,D2,D3, G, nodes_index=get_similarity_matrix(dotpath)
#print(D1.shape)
#graphname="collaboration"

#outdir="../html3Dviz/"
#pdb.set_trace()

#d = json_graph.node_link_data(G)
#json.dump(d, open(outdir+'graph_'+graphname+'.json', 'w'))
#j=open(outdir+'graph_'+graphname+'.json','w')
#j.write("edges='" + str(d['links']).replace("\'","\"") + "';" )

#j.write("nodes='" + str(d['nodes']).replace("\'","\"") + "';" )
#j.close()
