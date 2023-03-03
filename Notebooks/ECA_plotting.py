from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations, permutations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def normalize(vector, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(vector, order, axis))
    l2[l2==0] = 1
    return vector/np.expand_dims(l2, axis)[0]

def orderVerts(verts):
    #cetroid point of verts
    c = np.array([sum(x)/len(x) for x in list(zip(*verts))])
    
    #rays from centroid to eact vert
    v = verts-c
    magv = np.linalg.norm(v,axis=1)

    vnorm = v / magv[:, np.newaxis]
    
    #angle from arbitrary first ray
    #theta = np.arccos(v@np.atleast_2d(v[0]).T/magv/magv[0])
    theta = np.arccos(np.clip(vnorm@vnorm[0],-1,1)).T

    #determining alignment of subsequent rays
    ns = np.cross(vnorm[0],vnorm)
    nnorm = np.linalg.norm(ns,axis=1)
    ns = np.dot(ns,ns[np.argmax(nnorm)])
    
    
    #amending angle if >180 deg
    for i,th in enumerate(theta):
        if ns[i]<0:
            theta[i] = np.pi*2-th
    #sorting points 
    
    out = verts[theta.argsort(axis=0).squeeze()]
    return out,c
    
def drawCube(ax,r=[0,1]):
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot(*zip(s, e), color="k", ls = '--',lw=1)
    
def drawPlaneInCube(ax,pl,T,r=[0,1]):
    #T is offset value
    #pl is the normal vector of
    sfc = []
    pl = normalize(pl)
    
    for s, e in combinations(np.array(list(product(r, r, r))), 2): #loops through each pair of points
        
        if np.sum(np.abs(s-e)) == r[1]-r[0]: #checks if the pair of points are adjacent
            i = np.array([0,1,2]) 
            j = list(abs(s-e)).index(1) #determines the axis of the edges
            val = list(s)
            
            #chooses which indices are active
            i = i[i!=j]

            #calculates intercept at cube edge
            if pl[j]:
                val[j]= (T-pl[i[0]]*s[i[0]]-pl[i[1]]*s[i[1]])/pl[j]
            else:
                val[j] = -1000
            #keeps only plane within the unit cube
            if 0<=val[j]<=1:
                sfc.append(tuple(val))
    if sfc:           
        sfc = np.vstack([*set(sfc)])
        sfc,c = orderVerts(sfc)
        #ax.text(*c,f"{pl}") #plot the plane values
        pc = Poly3DCollection([sfc]) 
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        ax.add_collection3d(pc)

    return sfc 

def plotSetup(r = [0,1],fig= None,spdim = [1,1],i=1,proj = 'persp',elev=30., azim=330):
    if fig is None:
        fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(*spdim,i, projection='3d',proj_type = proj)
    s = 1.2
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
    # Get rid of the spines                         
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(r)
    ax.set_ylim(r)
    ax.set_zlim([0, s*r[1]])
    ax.set_xlabel('x1',labelpad=0)
    ax.set_ylabel('x2',labelpad=0)
    ax.set_zlabel('x3',labelpad=0)
    ax.set_box_aspect((1,1,s))
    ax.view_init(elev=elev, azim=azim)
    ax.zaxis._axinfo['juggled'] = (1,2,0)
    
    return fig,ax
def drawRule(ax,rule,r=[0,1],labelCorners=True):
    rulebin = f'{rule:#010b}'[-1:1:-1] #creates binary representation by rule number
    plt.title(f"Rule = {rule} : {rulebin}")
    rulebin = [bool(int(x)) for x in rulebin]
    ax.scatter(*list(zip(*np.array(list(product(r, r, r))))),c=rulebin,cmap='Greys',s=100,edgecolors='black',vmin=0,alpha=1)
    if labelCorners:
        for p in np.array(list(product(r, r, r))):
            ax.text(*p,f"{p[0]}{p[1]}{p[2]}\n",horizontalalignment='center',verticalalignment='bottom')
