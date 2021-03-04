### Compute dissimilarity matrices for the florentine families dataset ###

import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy

path = os.path.join(sys.path[0],
                    'florentine_families_relations_matrix_sample.csv')
with open(path) as csvfile:
    df = pd.read_csv(csvfile)

attributes = df.columns[2::] #attributes
attributes_count = np.sum(df.values[:,2::],0) #number of times attribute occurs
families,families_count = np.unique(df.values[:,0:2],return_counts=True)
families = list(families)

counts = pd.DataFrame(np.zeros((len(families),len(attributes))),
                      index=families, columns=attributes)
#number of times a given family has each attribute, as either actor
for i in df.index:
    name1 = df['actor1surname'][i]; name2 = df['actor2surname'][i]
    counts.loc[name1] += df.loc[i][attributes]
    counts.loc[name2] += df.loc[i][attributes]

def reduce_families(attributes_list,):
    """\
    Returns families that appear at least once in each of the attributes listed.
    
    --- arguments ---
    atrributes_list = list of attributes
    """
    for attribute in attributes_list:
        assert attribute in attributes

    fams = []
    for family in families:
        min_number = np.min(counts.loc[family][attributes_list])
        if min_number > 0:
            fams.append(family)
        
    return fams

def count_occurrences(attribute,allowed_families=None):
    assert attribute in attributes
    if allowed_families is None:
        allowed_families = families
        
    df0 = df[df[attribute]==1]

    families,families_count = np.unique(df0.values[:,0:2],return_counts=True)
    counts = []
    for family, count in zip(families,families_count):
        if family in allowed_families:
            i = allowed_families.index(family)
            counts.append(count)

    return counts

def similarity_graph(attribute,allowed_families=None):
    """\
    Returns similarity graph for given attribute and set of families.

    Parameters :

    attribute : string
    Name of attribute (e.g. 'marriage' or 'loan')

    allowed_families : list
    List of families. If not specified, it uses all the families (indexed as
    given by setup.families).

    Outputs:

    edges : list
    List containing edges, that is, pairs of indices of the nodes as given by
    allowed_families.

    counts : list
    List containing number of times each edge appears (of the same length as
    edges).
    """
    assert attribute in attributes
    if allowed_families is None:
        allowed_families = families
        
    df0 = df[df[attribute]==1]
    
    edges = []
    for i in df0.index:
        fam1 = df0['actor1surname'][i]
        fam2 = df0['actor2surname'][i]
        if fam1 in allowed_families and fam2 in allowed_families:
            j = allowed_families.index(fam1)
            k = allowed_families.index(fam2)
            edges.append([min(j,k),max(j,k)])
    
    edges, counts = np.unique(edges,return_counts=True,axis=0)

    return edges, counts

def setup_all(attributes_list):
    for attribute in attributes_list:
        assert attribute in attributes
    D = {}
    families_list = find_families(attributes_list)
    D['families'] = families_list
    D['graph'] = []
    for attribute in attributes_list:
        d = {}
        edges, counts = similarity_graph(attribute,families_list)
        d['label'] = attribute
        d['edges'] = edges
        d['counts'] = counts
        D['graph'].append(d)
    D['colors'] = []
    for attribute in attributes:
        d = {}
        counts = count_occurrences(attribute,families_list)
        d['label'] = attribute
        d['ncolor'] = counts
        D['colors'].append(d)
    return families_list, D['graph'], D['colors']

attributes2 = ['marriage','loan']
families2 = reduce_families(attributes2)

attributes3 = ['marriage','loan','business']
families3 = reduce_families(attributes3)

### OLD ###

def graphs(attributes_list,families_list):
    for attribute in attributes_list:
        assert attribute in attributes
    K = len(attributes_list)
    for family in families_list:
        assert family in families
    N = len(families_list)

    for i in df.index:
        fam1 = df['actor1surname'][i]
        fam2 = df['actor2surname'][i]
        if fam1 in families_list and fam2 in families_list:
            j = families_list.index(fam1)
            k = families_list.index(fam2)
            S[:,j,k] += df.loc[i][attributes_list]
            S[:,k,j] += df.loc[i][attributes_list]
    
def connections(attributes_list=['marriage','business','loan'],
                families_list=None,verbose=0):
    """\
    Returns tensor with number of connections between families for each 
    attribute. If families_list is None, it first finds the list of families
    containing at least one connection in each of the attributes listed.

    Returns:

    S : array (K x N x N)
    For attribute k and families i and j, S[k,i,j] is the number of connections
    between families i and j for attribute k.
    """
    for attribute in attributes_list:
        assert attribute in attributes
    K = len(attributes_list)
        
    if families_list is None:
        families_list = find_families(attributes_list,verbose=verbose)
    else:
        for family in families_list:
            assert family in families
    N = len(families_list)

    S = np.zeros((K,N,N))
    for i in df.index:
        fam1 = df['actor1surname'][i]; fam2 = df['actor2surname'][i]
        if fam1 in families_list and fam2 in families_list:
            j = families_list.index(fam1); k = families_list.index(fam2)
            S[:,j,k] += df.loc[i][attributes_list]
            S[:,k,j] += df.loc[i][attributes_list]

    if verbose > 0:
        SS = np.sum(S,axis=2)
        for i in range(N):
            print(families_list[i],SS[:,i])
    return S
        
def connected_components(attributes_list=['marriage','business','loan'],
                families_list=None):
    """\
    Returns connected components for each attribute and largest connected
    component common to all attributes.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if families_list is None:
        families_list = find_families(attributes_list)
    else:
        for family in families_list:
            assert family in families
    N = len(families_list)
    
    S = connections(attributes_list,families_list)
    K = len(attributes_list)

    largest_components = []
    largest_components_total = np.zeros(N)
    for k in range(K):
        graph = csr_matrix(S[k])
        n_components, labels = connected_components(csgraph=graph,
                                                    directed=False,
                                                    return_labels=True)
        print("k = ",k)
        print(labels)
        largest_component = []
        for n in range(N):
            if labels[n] == 0:
                largest_component.append(n)
                largest_components_total[n] += 1
        print(largest_component)
        largest_components.append(largest_component)

    overall_largest_component = []
    for n in range(N):
        if largest_components_total[n] == K:
            overall_largest_component.append(n)
    print(overall_largest_component)
    print([families_list[i] for i in overall_largest_component])

if __name__ == '__main__':
    attributes_list = attributes3
    families_list = find_families(attributes_list,verbose=1)
    #connections(attributes_list,families_list,verbose=1)
    #connections(attributes_list,verbose=1)
    #connected_components(attributes_list,families_list=families2)
    
    #S = similarity()
    #np.save('similarity_matrices.npy', S)
    #sys.path.append("../../mview")
    #import distance
    #D = distance.dmatrices(S,input_type='similarities')
    #np.save('distance_matrices.npy', D)
