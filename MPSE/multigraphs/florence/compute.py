import sys
import matplotlib.pyplot as plt
import numpy as np
import setup
sys.path.insert(1,'../../mview')
import distances, perspective, multiview, mds, compare, multigraph, mpse, mds2

attributes2 = ['marriage','loan']
families2 = ['Adimari', 'Ardinghelli', 'Arrigucci', 'Baldovinetti', 'Barbadori', 'Bardi', 'Bischeri', 'Brancacci', 'Busini', 'Castellani', 'Cavalcanti', 'Ciai', 'Corbinelli', 'Da Uzzano', 'Degli Agli', 'Del Forese', 'Della Casa', 'Fioravanti', 'Gianfigliazzi', 'Ginori', 'Giugni', 'Guadagni', 'Guicciardini', 'Lamberteschi', 'Manelli', 'Manovelli', 'Medici', 'Panciatichi', 'Pandolfini', 'Pazzi', 'Pecori', 'Peruzzi', 'Ricasoli', 'Rondinelli', 'Rossi', 'Salviati', 'Scambrilla', 'Serragli', 'Serristori', 'Spini', 'Strozzi', 'Tornabuoni']

attributes3 = ['marriage','loan','business']
families3 = ['Adimari', 'Ardinghelli', 'Arrigucci', 'Baldovinetti', 'Barbadori', 'Bardi', 'Bencivenni', 'Bischeri', 'Brancacci', 'Castellani', 'Cavalcanti', 'Da Uzzano', 'Della Casa', 'Guicciardini', 'Lamberteschi', 'Manelli', 'Manovelli', 'Medici', 'Orlandini', 'Panciatichi', 'Pazzi', 'Peruzzi', 'Ricasoli', 'Rondinelli', 'Rossi', 'Serragli', 'Serristori', 'Spini', 'Strozzi', 'Tornabuoni']

def example1():
    attribute = 'marriage'
    families = setup.find_families([attribute])
    S = setup.connections([attribute],families)[0]
    D_all = multigraph.sim2dist(S)
    colors = D_all[-2]

    D0 = multigraph.from_matrix(S,transformation='reciprocal')
    edges = D0['edges']
    
    mv = mds2.MDS(D0,verbose=1)
    mv.initialize()
    mv.stochastic()
    mv.agd(min_step=1e-6)
    mv.figureX(edges=True,colors=colors)
    mv.figureH()

    def f(x):
        if x <3:
            y = x
        else:
            y = None
        return y
    D = multigraph.from_matrix(D_all,transformation=f)
    mv = mds2.MDS(D,verbose=1)
    mv.initialize()
    mv.stochastic()
    mv.agd(min_step=1e-6)
    mv.figureX(edges=edges,colors=colors)
    mv.figureH()

    def f(x):
        if x <5:
            y = x
        else:
            y = None
        return y
    D = multigraph.from_matrix(D_all,transformation=f)
    mv = mds2.MDS(D,verbose=1)
    mv.initialize()
    mv.stochastic()
    mv.agd(min_step=1e-6)
    mv.figureX(edges=edges,colors=colors)
    mv.figureH()
    
    plt.show()
    
def example3a():
    attributes = attributes2
    families = families2
    
    S = setup.connections(attributes_list=attributes,families_list=families)
    D = []
    edges = []
    for s in S:
        edges.append(multigraph.from_matrix(s,transformation='reciprocal')\
                     ['edges'])
        d = multigraph.sim2dist(s)
        def f(x):
            if x <3:
                y = x
            else:
                y = None
            return y
        d = multigraph.from_matrix(d,transformation=f)
        D.append(d)
    K = len(S); N = len(S[0])

    p = perspective.Persp()
    p.fix_Q(number=K, special='standard')

    mv = mpse.MPSE(D,persp=p,verbose=1)
    mv.setup_visualization(visualization='mds')
    mv.initialize_X()
    mv.optimize_X(verbose=1)
    mv.figureX()
    mv.figureY(edges=edges)
    mv.figureH()
    plt.show()

def example2():
    attributes = attributes2
    families = families2
    
    S = setup.connections(attributes_list=attributes,families_list=families)
    D1 = dissimilarities.from_matrix(S[0],transformation='reciprocal')
    D2 = dissimilarities.from_matrix(S[1],transformation='reciprocal')
    D = [D1,D2]
   #D = distances.dmatrices(S,input_type='similarities',
    #                        connect_components=True,connect_factor=1.5)
    K = len(S); N = len(S[0])

    p = perspective.Persp()
    p.fix_Q(number=K, special='standard')
    
    a,b,c=compare.all(D,p,title='florence marriage + business',
                      names=attributes, edges=S, verbose=1)

    # distance to family with maximum number of  marriage & loan links
    colors = [D[0,-2],D[1,1]]
    compare.plot(a,b,c,title='florence marriage + loan / correct colors',
                 names=attributes, edges=S, colors=colors, verbose=1)

    # distance to family with maximum number of  marriage only
    #colors = [D[0,-2],D[0,-2]]
    #compare.plot(a,b,c,title='florence marriage + business',
     #            names=attributes, edges=S, colors=colors, verbose=1)

    # distance to family with maximum number of  marriage only
    #S0 = setup.connections(attributes_list=['business'],families_list=families)
    #D = distances.dmatrices(S0,input_type='similarities',
                   #         connect_components=True,connect_factor=1.5)
    #colors = [D[0,3],D[0,3]]
    #compare.plot(a,b,c,title='florence marriage + loan / business colors',
                # names=attributes, edges=S, colors=colors, verbose=1)

def example3():
    attributes = attributes3
    families = families3
    
    S = setup.connections(attributes_list=attributes,families_list=families)
    D = distances.dmatrices(S,input_type='similarities',
                            connect_components=True,connect_factor=1.5)
    K = len(S); N = len(S[0])

    p = perspective.Persp()
    p.fix_Q(number=K, special='standard')
    
    a,b,c=compare.all(D,p,title='florence marriage + business',
                      names=attributes, edges=S, verbose=1)

    # distance to family with maximum number of  marriage & loan links
    #colors = [D[0,-2],D[1,1],D[2,4]]
    colors = [D[0,-2],D[1,4],D[2,1]]
    compare.plot(a,b,c,title='florence marriage + business',
                 names=attributes, edges=S, colors=colors, verbose=1)

def example23():
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
    compare.plot(a,b,c,title='florence marriage + loan / correct colors',
                 names=attributes, edges=S, colors=colors, verbose=1)

    # distance to family with maximum number of  marriage only
    #colors = [D[0,-2],D[0,-2]]
    #compare.plot(a,b,c,title='florence marriage + business',
     #            names=attributes, edges=S, colors=colors, verbose=1)

    # distance to family with maximum number of  marriage only
    S0 = setup.connections(attributes_list=['business'],families_list=families)
    D = distances.dmatrices(S0,input_type='similarities',
                            connect_components=True,connect_factor=1.5)
    colors = [D[0,3],D[0,3]]
    compare.plot(a,b,c,title='florence marriage + loan / business colors',
                 names=attributes, edges=S, colors=colors, verbose=1)
    
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
        vis.graph(title=attributes[i])
        
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
        
    #proj = perspective.Persp()
    #proj.fix_Q(number=K, special='standard')

    #mv = multiview.Multiview(D,persp=proj,labels=families)
    #mv.setup_visualization()
    #mv.initialize_X()
    #mv.optimize_X(rate=0.002,max_iters=200)
    #mv.figureX(); mv.figureY();
    #mv.figure();
    #mv.graphY(k=0); mv.graphY(k=1)
    plt.show()

if __name__=='__main__':
    example1()
    #example3a()
    #example2()
    #example3()
    #compute_mds()
