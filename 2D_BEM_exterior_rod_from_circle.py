'''
Uses 2D BEM for finding the tangential component of the gradient of the solution of an exterior Laplace problem.
'''
import numpy as np
import matplotlib.pyplot as plt

def compute_v_prop_rod(M, cat, pos=None, constant_total_flux=False):
    '''
    Inputs:
    - M: determines number of discrete elements on surface (and thus accuracy)
    - cat: the fraction of the eight enzyme positions that are occupied
    - pos (optional): the positions of the occupied enzymes, default is next to each other,
    starting from the end of the rod
    - constant_total_flux (optional): if True, the total enzymatic catalytic activity is held constant,
    irrespective of how many enzyme spots are occupied. The catalytic activity per enzyme thus 
    decreases as the number of enzymes increases. Default is False.

    Outputs:
    - v_prop: simulated propulsion velocity vector. Also plots the normal and tangential components 
    of the concentration gradient at the micromotor surface.
    '''
    ############################## Inputs ##############################
    K = 1.
    D = 1.
    Ly = 10     # relative length of each vertical side
    ####################################################################

    Lx = Ly/10
    assert M%2==0, r'$M$ must be divisible by two'
    N = 11*M
    Nx = M//2       # number of surface parts for each horizontal side of the rod
    Ny = 5*M        # number of surface parts for each vertical side of the rod
    dxy = Ly/Ny     # spacing in horizontal and vertical direction: length of each surface part
    print(f'N = {N}, Nx = {Nx}, Ny = {Ny}')
    Ncat = int(np.floor(2*Ny*cat))
    if Ncat%2==0:
        pass
    else:
        Ncat += 1
    print(f'Ncat = {Ncat}, 2*Ny*cat (unrounded) = {2*Ny*cat}')

    du_dn = np.zeros(N, dtype=float)

    total_act = -K/D

    if constant_total_flux:
        act = total_act / (cat*8)   # local catalytic activity per enzyme
    else:   
        act = total_act             # local catalytic activity per enzyme

    if pos: # enzymes not next to each other lengthwise
        Npos = int(np.round(cat*8, 0))
        assert len(pos) == Npos, 'Number of positions given must match number of enzymes attached'
        Ncat = Ncat//(2*Npos)
        for i in pos:
            i-=1    # to 0-based
            du_dn[i*Ncat:(i+1)*Ncat] = act   # left-side enzymes
            du_dn[Ny+Nx+Ny-(i+1)*Ncat:Ny+Nx+Ny-i*Ncat] = act   # right-side enzymes
    else:
        du_dn[:Ncat//2] = act       # left-side enzymes
        du_dn[Ny+Nx+Ny-Ncat//2:Ny+Nx+Ny] = act 

    nx, ny = np.zeros_like(du_dn), np.zeros_like(du_dn)
    nx[:Ny] = 1.
    ny[Ny:Ny+Nx] = -1.
    nx[Ny+Nx:Ny+Nx+Ny] = -1.
    ny[Ny+Nx+Ny:] = 1.

    # Coordinates of line-piece centers:
    x_c = np.zeros_like(du_dn)
    y_c = np.zeros_like(du_dn)
    x_c[Ny:Ny+Nx] = np.linspace(dxy/2, Lx-dxy/2, Nx)
    x_c[Ny+Nx:Ny+Nx+Ny] = Lx
    x_c[Ny+Nx+Ny:] = np.linspace(Lx-dxy/2, dxy/2, Nx)
    y_c[:Ny] = np.linspace(dxy/2, Ly-dxy/2, Ny)
    y_c[Ny:Ny+Nx] = Ly
    y_c[Ny+Nx:Ny+Nx+Ny] = np.linspace(Ly-dxy/2, dxy/2, Ny)

    # Coordinates of line-piece beginnings and ends:
    x_e = np.zeros_like(du_dn)
    x_ee = np.zeros_like(du_dn)
    y_e = np.zeros_like(du_dn)
    y_ee = np.zeros_like(du_dn)
    x_e[Ny:Ny+Nx] = np.linspace(0, Lx-dxy, Nx)
    x_e[Ny+Nx:Ny+Nx+Ny] = Lx
    x_e[Ny+Nx+Ny:] = np.linspace(Lx, dxy, Nx)
    x_ee[Ny:Ny+Nx] = np.linspace(dxy, Lx, Nx)
    x_ee[Ny+Nx:Ny+Nx+Ny] = Lx
    x_ee[Ny+Nx+Ny:] = np.linspace(Lx-dxy, 0, Nx)
    y_e[:Ny] = np.linspace(0, Ly-dxy, Ny)
    y_e[Ny:Ny+Nx] = Ly
    y_e[Ny+Nx:Ny+Nx+Ny] = np.linspace(Ly, dxy, Ny)
    y_ee[:Ny] = np.linspace(dxy, Ly, Ny)
    y_ee[Ny:Ny+Nx] = Ly
    y_ee[Ny+Nx:Ny+Nx+Ny] = np.linspace(Ly-dxy, 0, Ny)

    x_, y_ = 0.5*(x_ee-x_e), 0.5*(y_ee-y_e)     # x_i^-, y_i^- in Supplementary Information

    'Tangent vector direction along each boundary element:'
    t_x, t_y = (x_ee-x_e)/dxy, (y_ee-y_e)/dxy

    H = np.zeros((N,N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i!=j:    # matrix elements equal zero otherwise
                H[i,j] = -1/np.pi * dxy/2 * 1/(2*(x_[j]**2 + y_[j]**2)) * ( 
                    (y_[j]*nx[j]-x_[j]*ny[j]) * 
                    (np.log(
                        (y_c[i]**2-2*1*(y_c[i]*y_[j]+x_c[i]*x_[j]-x_c[j]*x_[j]-y_c[j]*y_[j]) - 
                         2*y_c[i]*y_c[j] + y_c[j]**2 + x_c[i]**2 - 2*x_c[i]*x_c[j] + x_c[j]**2 + 
                         1**2*(y_[j]**2+x_[j]**2)))
                        - np.log(y_c[i]**2-2*(-1)*(y_c[i]*y_[j]+x_c[i]*x_[j]-x_c[j]*x_[j]-y_c[j]*y_[j]) - 
                                 2*y_c[i]*y_c[j] + y_c[j]**2 + x_c[i]**2 - 2*x_c[i]*x_c[j] + x_c[j]**2 + 
                                 (-1)**2*(y_[j]**2+x_[j]**2)) 
                    )
                    + 2*(y_[j]*ny[j] + x_[j]*nx[j]) * 
                    (np.atan2(
                        -y_c[i]*y_[j]+y_c[j]*y_[j]-x_c[i]*x_[j]+x_c[j]*x_[j]+y_[j]**2*1+x_[j]**2*1,
                        -y_c[i]*x_[j]+y_c[j]*x_[j]+x_c[i]*y_[j]-x_c[j]*y_[j]                     
                        ) - np.atan2(
                        -y_c[i]*y_[j]+y_c[j]*y_[j]-x_c[i]*x_[j]+x_c[j]*x_[j]+y_[j]**2*(-1)+x_[j]**2*(-1),
                        -y_c[i]*x_[j]+y_c[j]*x_[j]+x_c[i]*y_[j]-x_c[j]*y_[j]                      
                        )
                    )
                    )

    u_t = H@du_dn     # magnitude of tangential components of grad(u)

    v_prop = [-np.mean(u_t*t_x), -np.mean(u_t*t_y)]
    print(v_prop)

    plt.scatter(x_c, y_c, c=du_dn)
    plt.colorbar()

    plt.figure()
    plt.scatter(x_c, y_c, c=u_t)
    plt.colorbar()
    plt.show()
    return(v_prop)

# Results were generated using `M=1024`
v_prop = compute_v_prop_rod(M=1024, cat=2/8, pos=None, constant_total_flux=False)