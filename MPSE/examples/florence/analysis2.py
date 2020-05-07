import sys
import matplotlib.pyplot as plt
import numpy as np
import setup
sys.path.insert(1,'../..')
import mview
import setup

attributes2 = ['marriage','loan']

families = setup.families
families2 = setup.reduce_families(attributes2)
bad_indices = [32,21,17,11,7,6]
for bad_index in bad_indices:
    del families2[bad_index]

#with open('names2.txt', 'w') as filehandle:
#    for family in families2:
#        filehandle.write(f'{family}\n')
    
marriage_count = setup.count_occurrences('marriage',families2)
loan_count = setup.count_occurrences('loan',families2)

marriage_edges, marriage_counts = \
setup.similarity_graph('marriage',allowed_families=families2)
marriage_dissimilarities = 1.0/np.array(marriage_counts)

loan_edges, loan_counts = \
setup.similarity_graph('loan',allowed_families=families2)
loan_dissimilarities = 1.0/np.array(loan_counts)

func = lambda x : 1.0/x
index = 41

diss = mview.DISS(len(families2),node_labels=families2)
diss.add_graph(edge_list=marriage_edges,
               dissimilarity_list=marriage_dissimilarities,
               shortest_path=True,weight_function=func,label='marriage',
               node_colors=index)
diss.add_graph(edge_list=loan_edges,dissimilarity_list=loan_dissimilarities,
               shortest_path=True,weight_function=func,label='loan',
               node_colors=index)

def mds1(weighted=True,**kwargs):
    print()
    mds1 = mview.MDS(diss.D[0],weighted=weighted,verbose=2)
    mds1.gd(**kwargs)
    mds1.figureH('mds - marriage')
    mds1.figureX(edges=marriage_edges,labels=True,axis=False,markersize=50,
                 title=None)

def mds2(weighted=True,**kwargs):
    print()
    mds2 = mview.MDS(diss.D[1],weighted=weighted,verbose=2)
    mds2.gd(**kwargs)
    mds2.figureH('mds - loan')
    mds2.figureX(edges=loan_edges,labels=True,axis=False,markersize=50,
                 title='')
    
def mpse(weighted=True,**kwargs):
    print()
    mv = mview.MPSE(diss,weighted=weighted,verbose=2)
    mv.gd(min_grad=1e-4,max_iter=300,lr=0.1,average_neighbors=1)
    mv.gd(min_grad=1e-4,max_iter=200,lr=[2,0.01],scheme='fixed')
    mv.figureH()
    mv.figureX(edges=marriage_edges,axis=True,labels=True,markersize=50,
               colors=None)
    mv.figureX(edges=loan_edges,axis=True,labels=True,markersize=50,colors=None)
    mv.figureX(axis=True,labels=True,markersize=50,colors=None)
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    axes=[ax1,ax2]
    mv.figureY(include_colors=True,
               edges=[marriage_edges,loan_edges],labels=True,axis=False,
               title=None,markersize=50,ax=axes)
    plt.draw()
    np.savetxt('X.csv', mv.X, delimiter=',')
    np.savetxt('Q1.csv', mv.Q[0], delimiter=',')
    np.savetxt('Q2.csv', mv.Q[1], delimiter=',')
    
if __name__=='__main__':
    #mds1(max_iter=100,weighted=True)
    #mds2(weighted=True)
    #mpse(weighted=True)
    #plt.show()

#    print(', '.join([family for family in families2]))
#    print(', '.join([family+f'({i+1})' for i,family in zip(range(len(families2)),families2)]))
    print()
