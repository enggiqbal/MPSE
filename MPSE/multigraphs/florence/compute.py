import sys
import matplotlib.pyplot as plt
import numpy as np
import setup
sys.path.insert(1,'../../mview')
import distances, perspective, multiview, mds, compare

attributes2 = ['marriage','loan']
families2 = ['Adimari', 'Ardinghelli', 'Arrigucci', 'Baldovinetti', 'Barbadori', 'Bardi', 'Bischeri', 'Brancacci', 'Busini', 'Castellani', 'Cavalcanti', 'Ciai', 'Corbinelli', 'Da Uzzano', 'Degli Agli', 'Del Forese', 'Della Casa', 'Fioravanti', 'Gianfigliazzi', 'Ginori', 'Giugni', 'Guadagni', 'Guicciardini', 'Lamberteschi', 'Manelli', 'Manovelli', 'Medici', 'Panciatichi', 'Pandolfini', 'Pazzi', 'Pecori', 'Peruzzi', 'Ricasoli', 'Rondinelli', 'Rossi', 'Salviati', 'Scambrilla', 'Serragli', 'Serristori', 'Spini', 'Strozzi', 'Tornabuoni']

attributes3 = ['marriage','business','loan']
#families3 = ['Adimari', 'Ardinghelli', 'Baldovinetti', 'Bardi', 'Brancacci', 'Castellani', 'Cavalcanti', 'Da Uzzano', 'Della Casa', 'Guicciardini', 'Manelli', 'Manovelli', 'Rondinelli', 'Rossi', 'Serragli', 'Spini']
families3 = setup.find_families(attributes3)

def example2():
    attributes = attributes2
    families = families2
    
    S = setup.connections(attributes_list=attributes,families_list=families)
    D = distances.dmatrices(S,input_type='similarities',
                            connect_components=True,connect_factor=1.5)
    K = len(S); N = len(S[0])

    p = perspective.Persp()
    p.fix_Q(number=K, special='standard')
    
    a,b,c=compare.all(D,p,title='florence marriage + business',
                      names=attributes, edges=S, verbose=1)

    # distance to family with maximum number of  marriage & loan links
    colors = [D[0,-2],D[1,1]]
    compare.plot(a,b,c,title='florence marriage + business',
                 names=attributes, edges=S, colors=colors, verbose=1)

def example3():
    attributes = attributes3
    families = families3
    
    S = setup.connections(attributes_list=attributes,families_list=families)
    D = distances.dmatrices(S,input_type='similarities',
                            connect_components=True,connect_factor=1.5)
    K = len(S); N = len(S[0])

    p = perspective.Persp()
    p.fix_Q(number=K, special='standard')
    
    compare.main(D,p,title='florence marriage + business',
                 names=attributes, edges=S, verbose=1)
    
def compute_mds(num=2):
    if num==2:
        attributes = attributes2
        families = families2
    elif num==3:
        attributes = attributes3
        families = families3
        
    S = setup.connections(attributes_list=attributes,families_list=families)
    D = distances.dmatrices(S,input_type='similarities',
                            connect_components=True,connect_factor=1.5)
    K = len(S); N = len(S[0])
    
    fig, axs = plt.subplots(1,K,sharey=True,sharex=True)
    for i in range(K):
        vis = mds.MDS(D[i],labels=families)
        vis.initialize()
        vis.optimize(algorithm='agd')
        vis.graph()
        
        axs[i].title.set_text(attributes[i])
        if i ==0 :
            for n in range(N):
                axs[i].scatter(vis.X[n,0],vis.X[n,1],label=families[n])
        else:
            for n in range(N):
                axs[i].scatter(vis.X[n,0],vis.X[n,1])
        fig.legend(loc=7)
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)   
        plt.show(block=False)
        fig = vis.figure(); plt.show(block=False)
        
    proj = perspective.Persp()
    proj.fix_Q(number=K, special='standard')

    mv = multiview.Multiview(D,persp=proj,labels=families)
    mv.setup_visualization()
    mv.initialize_X()
    mv.optimize_X(rate=0.002,max_iters=200)
    mv.figureX(); mv.figureY();
    mv.figure();
    mv.graphY(k=0); mv.graphY(k=1)
    plt.show()

if __name__=='__main__':
    example2()
    #example3()
