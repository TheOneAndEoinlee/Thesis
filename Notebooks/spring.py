import numpy as np

def spring(xa, ya, xb, yb, *args):
    # SPRING         Calculates the position of a 2D spring
    #    [XS YS] = SPRING(XA,YA,XB,YB,NE,A,R0) calculates the position of
    #    points XS, YS of a spring with ends in (XA,YA) and (XB,YB), number
    #    of coils equal to NE, natural length A, and natural radius R0. 
    #    Useful for mass-spring oscillation animations.
    # USAGE: in a first call in your code, call it with the full parameters.
    # Then, only you have to give it the coordinates of the ends.
    # EXAMPLE:
    # xa = 0; ya = 0; xb = 2; yb = 2; ne = 10; a = 1; ro = 0.1;
    # [xs,ys] = spring(xa,ya,xb,yb,ne,a,ro); plot(xs,ys,'LineWidth',2)
    #...
    # [xs,ys]=spring(xa,ya,xb,yb); plot(xs,ys,'LineWidth',2)
    #
    #   Made by:            Gustavo Morales   UC  08-17-09 gmorales@uc.edu.ve

    global ne, Li_2, ei, b

    if len(args) > 2: # calculating some fixed spring parameters only once time
        ne, a, r0 = args[0:3]                  # ne: number of coils - a = natural length - r0 = natural radius
        Li_2 =  (a/(4*ne))**2 + r0**2                # (large of a quarter of coil)^2
        ei = np.arange(2*ne+2)                            # vector of longitudinal positions
        j = np.arange(2*ne) 
        b = np.concatenate(([0], (-1)**j, [0])) # vector of transversal positions

    R = np.array([xb, yb]) - np.array([xa, ya])
    mod_R = np.linalg.norm(R) # relative position between "end_B" and "end_A"
    L_2 = (mod_R/(4*ne))**2 # (actual longitudinal extensión of a coil )^2
    if L_2 > Li_2:
        raise ValueError("Initial conditions cause pulling the spring beyond its maximum large. Try reducing these conditions.")
    else:
        r = np.sqrt(Li_2 - L_2)   #actual radius
    c = r*b    # vector of transversal positions
    u1 = R/mod_R 
    u2 = np.array([-u1[1], u1[0]]) # unitary longitudinal and transversal vectors 
    xs = xa + u1[0]*(mod_R/(2*ne+1))*ei + u2[0]*c # horizontal coordinates
    ys = ya + u1[1]*(mod_R/(2*ne+1))*ei + u2[1]*c # vertical coordinates
    
    return xs, ys
