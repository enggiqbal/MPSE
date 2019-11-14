import gmds

def main(D1,D2,D3,feedback=False):
    """\
    Run multiview-MDS algorithm, optimizing for both data positions and
    projections.

    --- arguments ---
    D1, D2, D3 = distance matrices for each of three projections
    - each distance matrix is an (n x n) array

    feedback = print progress and intermediate results if set to True

    --- outputs ---
    X0 = initial positions obtained from MDS iteration (n x n array)
    Q0 = initial
    """
     

    X,Qs,stress,X0,Q0s = gmds.special.optimal_XQ([D1,D2,D3],feedback=feedback)
    points = X; points0 = X0
    proj = gmds.special.compute_projections(Qs)
    proj0 = gmds.special.compute_projections(Q0s)
    cost = stress[-1]
    costhistory = stress
    return points, proj, cost, costhistory, Qs, X0, proj0, Q0s
