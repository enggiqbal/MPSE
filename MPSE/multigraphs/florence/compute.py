import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1,'../../mview')
import distances, projections, multiview, mds

def compute_mds():
    S = np.load('similarity_matrices.npy')
    D = distances.dmatrices(S,input_type='similarities',connect_factor=2.0)

    for i in range(3):
        vis = mds.MDS(D[i])
        vis.initialize(params=10)
        vis.optimize(rate=0.005,max_iters=50)
        fig = vis.figure(); plt.show(block=False)
        
    proj = projections.Proj()
    proj.set_params_list(number=3, special='standard')

    mv = multiview.Multiview(D,proj=proj)
    mv.setup_technique('mds')
    mv.initialize_X()
    mv.optimize_X(algorithm='gd',rate=0.002,max_iters=200)
    mv.figureX(); mv.figureY(); plt.show()

if __name__=='__main__':
    compute_mds()
