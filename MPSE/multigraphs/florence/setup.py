### Compute dissimilarity matrices for the florentine families dataset ###

import sys
import numpy as np
import pandas as pd
import networkx as nx

with open('florentine_families_relations_matrix_sample.csv') as csvfile:
    df = pd.read_csv(csvfile)

attributes = df.columns[2::] #attributes
attributes_count = np.sum(df.values[:,2::],0) #number of times attribute occurs
families,families_count = np.unique(df.values[:,0:2],return_counts=True)

counts = pd.DataFrame(np.zeros((len(families),len(attributes))),
                      index=families, columns=attributes)
#number of times a given family has each attribute, as either actor
for i in df.index:
    name1 = df['actor1surname'][i]; name2 = df['actor2surname'][i]
    counts.loc[name1] += df.loc[i][attributes]
    counts.loc[name2] += df.loc[i][attributes]

def find_families(attributes_list=['marriage','business','loan']):
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

def neighbors_number(attributes_list=['marriage','business','loan']):
    """\
    Print number of neighbors for each family
    """
    fams = find_families(attributes_list); N = len(fams)
    count = np.zeros((N,K))
    for i in df.index:
        fam1 = df['actor1surname'][i]; fam2 = df['actor2surname'][i]
        if fam1 in fams

def similarity(attributes_list=['marriage','business','loan'],
               symmetric=True):
    """\
    """
    K = len(attributes_list)
    fams = find_families(attributes_list); N = len(fams)
    S = np.zeros((K,N,N))
    for i in df.index:
        fam1 = df['actor1surname'][i]; fam2 = df['actor2surname'][i]
        if fam1 in fams and fam2 in fams:
            j = fams.index(fam1); k = fams.index(fam2)
            S[:,j,k] += df.loc[i][attributes_list]
            if symmetric is True:
                S[:,k,j] += df.loc[i][attributes_list]
    return S

if __name__ == '__main__':
    S = similarity()
    np.save('similarity_matrices.npy', S)
    sys.path.append("../../mview")
    import distance
    D = distance.dmatrices(S,input_type='similarities')
    np.save('distance_matrices.npy', D)
