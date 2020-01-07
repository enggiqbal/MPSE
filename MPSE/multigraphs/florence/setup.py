### Compute dissimilarity matrices for the florentine families dataset ###

import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy

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

attributes2 = ['marriage','loan']
attributes3 = ['marriage','loan','business']

families2 = ['Adimari', 'Ardinghelli', 'Arrigucci', 'Baldovinetti', 'Barbadori', 'Bardi', 'Bischeri', 'Brancacci', 'Busini', 'Castellani', 'Cavalcanti', 'Ciai', 'Corbinelli', 'Da Uzzano', 'Degli Agli', 'Del Forese', 'Della Casa', 'Fioravanti', 'Gianfigliazzi', 'Ginori', 'Giugni', 'Guadagni', 'Guicciardini', 'Lamberteschi', 'Manelli', 'Manovelli', 'Medici', 'Panciatichi', 'Pandolfini', 'Pazzi', 'Pecori', 'Peruzzi', 'Ricasoli', 'Rondinelli', 'Rossi', 'Salviati', 'Scambrilla', 'Serragli', 'Serristori', 'Spini', 'Strozzi', 'Tornabuoni']

def find_families(attributes_list=['marriage','business','loan'],verbose=0):
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

    if verbose > 0:
        print('- florence.setup.find_families():')
        print('  attributes =',attributes_list)
        print('  families =',fams)
        
    return fams

def connections(attributes_list=['marriage','business','loan'],
                families_list=None,verbose=0):
    """\
    Returns tensor with number of connectiosn between families for each 
    attribute. If families_list is None, it first finds the list of families
    containing at least one connection in each of the attributes listed.
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
    families_list = families2
    #find_families(attributes_list,verbose=1)
    #connections(attributes_list,families_list,verbose=1)
    connections(attributes_list,verbose=1)
    #connected_components(attributes_list,families_list=families2)
    
    #connected_components(families_list=['Adimari', 'Ardinghelli', 'Arrigucci', 'Baldovinetti', 'Barbadori', 'Bardi', 'Bischeri', 'Brancacci', 'Castellani', 'Cavalcanti', 'Da Uzzano', 'Della Casa', 'Guicciardini', 'Manelli', 'Manovelli', 'Medici', 'Panciatichi', 'Peruzzi', 'Ricasoli', 'Rondinelli', 'Rossi', 'Serragli', 'Serristori', 'Spini', 'Strozzi', 'Tornabuoni'])
    
    #S = similarity()
    #np.save('similarity_matrices.npy', S)
    #sys.path.append("../../mview")
    #import distance
    #D = distance.dmatrices(S,input_type='similarities')
    #np.save('distance_matrices.npy', D)
