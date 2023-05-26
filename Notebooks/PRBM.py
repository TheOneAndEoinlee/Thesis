import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

l1,l2,h1,h2,d1,d2,k_theta,k_phi,k_L, x1,x2,x3 = sm.symbols("l1,l2,h1,h2,d1,d2,k_theta,k_phi,k_L, x1,x2,x3", positive=True, real=True)



L1 = sm.sqrt(l1**2 + h1**2)
th0 = sm.atan(h1/l1)
th = sm.atan((h1-d1)/l1)

d2 = 2*(sm.sqrt(l1**2+h1**2-(h1-d1)**2)-l1)
ph0 = sm.atan(h2/l2)
ph = sm.atan((h2-d2)/l2)

d3 = sm.sqrt(l2**2+h2**2-(h2-d2)**2)-l2

L2 = sm.sqrt(l2**2+h2**2)

PE1 = 1/2*k_theta*(th0-th)**2
PE2 = 1/2*k_phi*(ph0-ph)**2
PE3 = 1/2*k_L*(d3)**2

PE = 8*PE1+8*PE2+2*PE3

c = sm.cos
s = sm.sin

F1 = PE1.diff(d1)
F2 = PE2.diff(d1)
F3 = PE3.diff(d1)
F = PE.diff(d1)
xs = [0,L1*c(th),2*L1*c(th),2*l1+h2,2*L1*c(th),2*l1+h2]
ys = [0,h1-d1,0,l2+d3,0,-l2-d3]

dF = F.diff(d1)


#convert F and PE to numpy functions with the parameters as input
F_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],F)
F1_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],F1)
F2_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],F2)
F3_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],F3)

PE_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],PE)
PE1_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],PE1)
PE2_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],PE2)
PE3_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],PE3)
xs_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],xs)
ys_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],ys)
L1_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],L1)
L2_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],L2)
dF_l = sm.lambdify([l1,l2,h1,h2,k_theta,k_phi,k_L,d1],dF)

#parameters
repl = {'l1':1.4,
        'l2':0.4,
        'h1':0.5,
        'h2':0.1,
        'k_theta':1,
        'k_phi':1,
        'k_L':6000}


def evaluate(l1=10,l2=0.4,h1=4,h2=0.8,k_theta=1,k_phi=1,k_L=1,d1 =0):
    #calls the lambdified functions with the parameters as input

    # TODO remove hard coded parameter multiplication for PE1, PE2, PE3
    Fl = F_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    F1l = 8*F1_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    F2l = 8*F2_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    F3l = 2*F3_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)

    PEl = PE_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    PE1l = 8*PE1_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    PE2l = 8*PE2_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    PE3l = 2*PE3_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    xs = xs_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    ys = ys_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)
    dF = dF_l(l1,l2,h1,h2,k_theta,k_phi,k_L,d1)

    return {'F':Fl,'PE':PEl,'PE1':PE1l,'PE2':PE2l,'PE3':PE3l,'x':xs,'y':ys,'dF':dF,'F1':F1l,'F2':F2l,'F3':F3l}



import scipy.optimize as opt
import numpy as np

#define function to fine all roots of  a function in a given range
#sample n initial guesses between a and b
def find_roots(f,a,b,n=100,df = None):
    #sample n initial guesses between a and b
    x0 = np.linspace(a,b,n)
    #find roots
    roots = set()
    for x in x0:
        try:
            r = opt.newton(f,x,fprime=df)
            roots.add(r.round(5))
        except:
            pass
    return roots



def evalgeom(repl,lam=1/6,gamma=1/5,h1_scale=12.5, t =0.8, kL= 3):

    #evaluates the geometry of the mechanism for a given set of design parameters

    #rescale repl based on h1 scale
    repl['l1'] = repl['l1']*h1_scale/repl['h1']
    repl['l2'] = repl['l2']*h1_scale/repl['h1']
    repl['h2'] = repl['h2']*h1_scale/repl['h1']
    repl['h1'] = h1_scale
    

    #calculate the length of the link
    Lp1 = L1_l(**repl)
    Lp2 = L2_l(**repl)

    
    
    #calculate the length of the spring
    Ls_1 = lam*(1-lam)*Lp1
    Ls_2 = lam*(1-lam)*Lp2

    #calculate the length of the reinfocement
    LRF_1 = Lp1 - Ls_1
    LRF_2 = Lp2 - Ls_2

    T= t/gamma

    #creat dict with the lengths of spring and reinforcement
    Ls = {'Ls_1':Ls_1,'Ls_2':Ls_2,'T':T,'t':t,"kL":kL}

    geom = {"L1":repl["l1"],"L2":repl["l2"],
        "H1":repl["h1"],"H2":repl["h2"],
        "fl1":Ls["Ls_1"],"fl2":Ls["Ls_2"],'Trf':Ls['T'],'t':Ls['t'],
        'kL':Ls['kL']}

    return geom

def spring(xa, ya, xb, yb, ne=None, a=None, r0=None):
    global Li_2, ei, b
    
    if ne is not None and a is not None and r0 is not None:
        # calculating some fixed spring parameters only once time
        Li_2 = (a/(4*ne))**2 + r0**2  # (large of a quarter of coil)^2
        ei = np.arange(2*ne+1)        # vector of longitudinal positions
        j = np.arange(2*ne) 
        b = np.append([0], (-1)**j)   # vector of transversal positions
    
    R = np.array([xb, yb]) - np.array([xa, ya]) # relative position between "end_B" and "end_A"
    mod_R = np.linalg.norm(R)
    
    L_2 = (mod_R/(4*ne))**2  # (actual longitudinal extensión of a coil )^2
    if L_2 > Li_2:
        raise ValueError("Initial conditions cause pulling the spring beyond its maximum large. Try reducing these conditions.")
    else:
        r = np.sqrt(Li_2 - L_2)  # actual radius
    
    c = r * b   # vector of transversal positions
    u1 = R / mod_R
    u2 = np.array([-u1[1], u1[0]])  # unitary longitudinal and transversal vectors
    
    xs = xa + u1[0] * (mod_R / (2*ne+1)) * ei + u2[0] * c
    ys = ya + u1[1] * (mod_R / (2*ne+1)) * ei + u2[1] * c
    
    return xs, ys

# geom = evalgeom(lam, repl, h1_scale=10, t=0.8, kL = 3)

# geom
