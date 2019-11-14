###########################################

### MDSp optimization ###

def MDSp_stress(X,P,D):
    """\
    Returns s2(X*P.T;D)

    X : data positions (n x p)
    P : transformation matrix (p x p)
    D : target distances (n x n)
    """
    return MDS_stress(X @ P.T,D)

def MDSp_Xgradient(X,P,D):
    """\
    Gradient of s2(X*P.T;D) w.r.t. X

    X : data positions (n x p)
    P : transformation matrix (p x p)
    D : target distances (n x n)
    """
    return MDS_gradient(X @ P.T, D) @ P

def MDSp_Pgradient(X,P,D):
    """\
    Gradient of s2(XP^T;D) w.r.t. P

    X : data positions (n x p)
    P : transformation matrix (p x p)
    D : target distances (n x n)
    """
    return MDS_gradient(X @ P.T, D).T @ X

def MDSp_Pdescent(X,D,P0,feedback=False,
                trajectory=False,**kwargs):
    """\
    Gradient descent on s2(X*P.T;D) w.r.t. P

    X : data positions (n x p)
    D : target distances (n x n)
    P0 : initial transformation matrix (p x p)

    feedback : return feedback if True
    trajectory : return trajectory if True

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion
    """
    if feedback is True:
        print("Beginning MDSp P descent")
        print("initial stress = ", MDSp_stress(X,P0,D))
        
    dP = lambda P: MDSp_Pgradient(X,P,D)

    if trajectory is True:
        P = gd.gradient_descent_trajectory(dP,P0,**kwargs)
    else:
        P = gd.gradient_descent(dP,P0,**kwargs)
    
    if feedback is True:
        if trajectory is True:
            P_final = P[-1]
        else:
            P_final = P
        print("final stress = ",MDSp_stress(X,P_final,D))
        
    return P

### Multiview MDSp optimization ###

def mMDSp_stress(X,PP,DD):
    """\
    Returns multiMDS stress

    X : data positions (n x p)
    PP : list of transformation matrices (k x p x p)
    DD : list of target distance matrices (k x n x n)
    """
    K = len(PP)
    stress = 0
    for k in range(K):
        stress += MDS_stress(X @ PP[k].T,DD[k])
    return stress

def mMDSp_Xgradient(X,PP,DD):
    """\
    Returns multiview-MDS stress

    X : data positions (n x p)
    PP : list of transformation matrices (k x p x p)
    DD : list of target distance matrices (k x n x n)
    """
    K = len(PP)
    dX = np.zeros(X.shape)
    for k in range(K):
        dX += MDSp_Xgradient(X,PP[k],DD[k])
    return dX

def mMDSp_Xdescent(X0,PP,DD,feedback=False,trajectory=False,**kwargs):
    """\
    Gradient descent for multiview-MDS with fixed transformations

    --- arguments ---
    X0 : initial data positions (n x p)
    PP : list of transformation matrices (k x p x p)
    DD : list of target distance matrices (k x n x n)

    feedback : return feedback if True
    trajectory : return trajectory if True

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion
    """
    if feedback is True:
        print("Beginning multiMDSp X descent")
        print("initial stress = ", mMDSp_stress(X0,PP,DD))
        
    dX = lambda X: mMDSp_Xgradient(X,PP,DD)

    if trajectory is True:
        X = gd.gradient_descent_trajectory(dX,X0,**kwargs)
    else:
        X = gd.gradient_descent(dX,X0,**kwargs)

    if feedback is True:
        if trajectory is True:
            X_final = X[-1]
        else:
            X_final = X
        print(" final stress = ",mMDSp_stress(X_final,PP,DD))
    return X

def mMDSp_XPdescent(X0,PP0,DD,feedback=False,**kwargs):
    """\
    Multiview-MDS optimization for X and P using gradient descent 

    X0 : initial data positions (n x p)
    PP0 : list of initial maps (k x p x p)
    DD : list of target distance matrices (k x n x n)

    feedback : return feedback if True
    trajectory : return trajectory if True

    --- kwargs ---
    rate : learning rate
    precision : stoping criterion
    """
    if feedback is True:
        print("\nBeginning multiview-MDSp XP descent")
        print("initial stress = ", mMDSp_stress(X0,PP0,DD))

    XX0 = [X0]+PP0 #list containing initial data for all coordinates
    dXX = []
    dX = lambda XX: mMDSp_Xgradient(XX[0],XX[1::],DD)
    dXX += [dX]
    K = len(PP0);
    for k in range(K):
        dPk = lambda XX: MDSp_Pgradient(XX[0],XX[1+k],DD[k])
        dXX += [dPk]

    XX = gd.coord_gradient_descent(dXX,XX0,**kwargs)
    X = XX[0]
    PP = XX[1::]
    
    if feedback is True:
        print("final stress = ",mMDSp_stress(X,PP,DD))
        
    return X,PP
