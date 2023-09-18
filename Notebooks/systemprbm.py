import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sig
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import prbm_helper_functions as phf



class UnitCell:
    """
    A class representing a unit cell in the system.

    Attributes:
    - alpha (float): the angle of the left leg with respect to the vertical axis
    - theta (float): the angle of the right leg with respect to the vertical axis
    - params (dict): a dictionary containing the mechanism parameters
    - i (int): the index of the unit cell in the system
    - left_neighbor (UnitCell): the left neighbor of the unit cell
    - right_neighbor (UnitCell): the right neighbor of the unit cell
    - F_d_func (function): a function that computes the force residual in the left leg
    - F_b_func (function): a function that computes the force residual in the right leg
    """
    def __init__(self, alpha_init, theta_init, params,i):
        self.alpha = alpha_init
        self.theta = theta_init
        self.params = params  
        self.i = i  


        self.left_neighbor: UnitCell = None
        self.right_neighbor: UnitCell = None

        self.define_symbols_and_parameters()

        self.expressions = self.define_symbolic_functions()
        
        self.create_lambda_functions()


    def define_symbols_and_parameters(self):
        self.alpha_sym, self.theta_sym, self.u_sym, self.theta_prev_sym, self.theta_next_sym = sm.symbols('alpha, theta, u, theta_prev, theta_next')
        self.params_symbols = {
            'alpha0': sm.symbols('alpha0'),
            'beta0': sm.symbols('beta0'),
            'gamma0': sm.symbols('gamma0'),
            'theta0': sm.symbols('theta0'),
            'L1': sm.symbols('L1'),
            'L2': sm.symbols('L2'),
            'L3': sm.symbols('L3'),
            'L4': sm.symbols('L4'),
            'C1': sm.symbols('C1'),
            'C2': sm.symbols('C2'),
            'C3': sm.symbols('C3'),
            'C4': sm.symbols('C4'),
            'Kf1': sm.symbols('Kf1'),
            'Kf2': sm.symbols('Kf2'),
            'Kf3': sm.symbols('Kf3'),
            'Kf4': sm.symbols('Kf4'),
            'K1': sm.symbols('K1'),
            'K2': sm.symbols('K2'),
            'K3': sm.symbols('K3'),
            'tog_offset': sm.symbols('tog_offset'),
        }

    def define_symbolic_functions(self):
        
        # Extract symbols
        alpha = self.alpha_sym
        theta = self.theta_sym
        u = self.u_sym
        theta_prev = self.theta_prev_sym
        theta_next = self.theta_next_sym
        params = self.params_symbols.copy()
        params.update(self.params) # Merge with numerical parameters

        alpha0 = params['alpha0']
        beta0 = params['beta0']
        gamma0 = params['gamma0']
        theta0 = params['theta0']
        L1 = params['L1']
        L2 = params['L2']
        L3 = params['L3']
        L4 = params['L4']
        C1 = params['C1']
        C2 = params['C2']
        C3 = params['C3']
        C4 = params['C4']
        Kf1 = params['Kf1']
        Kf2 = params['Kf2']
        Kf3 = params['Kf3']
        Kf4 = params['Kf4']
        K1 = params['K1']
        K2 = params['K2']
        K3 = params['K3']
        tog_offset = params['tog_offset']


        d0 = L1*(sm.sin(alpha0) - sm.sin(alpha)) #horizontal displacement of the decision element end effector
        d1 = 2*L1*(sm.cos(alpha) - sm.cos(alpha0)) #vertical displacement of the decision element bistable shuttle
        beta = sm.asin((L2*sm.sin(beta0)-d1)/L2) #angle of bistable shuttle with respect to the horizontal axis
        d2 = (L2*sm.sin(beta0)-d1)/(sm.tan(beta))-L2*sm.cos(beta0) #horizontal displacement of the bistable support
        gamma = sm.asin((L3*sm.sin(gamma0)-d1)/L3) #angle of the signal router with respect to the horizontal axis
        d3 = (L3*sm.sin(gamma0)-d1)/(sm.tan(gamma))-L3*sm.cos(gamma0)  #horizontal displacement of the signal router
        L5 = (L4*sm.cos(theta0)-(L1*sm.cos(alpha0) + #lever arm of the signal on the bifurcation element
              L3*sm.sin(gamma0)))/sm.cos(theta0)
        d4 = L5*(sm.sin(theta0)-sm.sin(theta)) #horizontal displacement of the signal on the bifurcation element
        d5 = 2*L4*(sm.cos(theta) - sm.cos(theta0))+u #vertical displacement of the bifurcation element input point
        d6 = L4*(sm.sin(theta0) - sm.sin(theta)) #horizontal displacement of the bifurcation element end effector
        d6_prev = L4*(sm.sin(theta0) - sm.sin(theta_prev)) #horizontal displacement of the bifurcation element end effector of the previous cell
        d6_next = L4*(sm.sin(theta0) - sm.sin(theta_next)) #horizontal displacement of the bifurcation element end effector of the next cell

        self.L5 = L5

        #define offset to introduce slack in the system
        K1_tog = sm.Heaviside(d6_prev-d0 - tog_offset)
        K2_tog = sm.Heaviside(d6-d0 - tog_offset)
        K3_tog = sm.Heaviside(d6_next-d0 - tog_offset)
        C4_tog = sm.Heaviside(d5)

        # define the energy of the system
        E = 0.5*(8*Kf1*(alpha-alpha0)**2
                 + 8*Kf2*(beta-beta0)**2
                 + 2*Kf3*(gamma-gamma0)**2
                 + 8*Kf4*(theta-theta0)**2
                 + C1*(d2)**2
                 + C2*(d3)**2
                 + C3*(d4-d3)**2
                 + C4*C4_tog*(d5)**2
                 + K1*K1_tog*(d6_prev-d0-tog_offset)**2
                 + K2*K2_tog*(d6-d0-tog_offset)**2
                 + K3*K3_tog*(d6_next-d0-tog_offset)**2
                 )

        # define the force of the system
        #This is a bug, this is the moment of the force, since alpha is the angle of the left leg
        #this explains the other bug!!!
        M_d = sm.diff(E, alpha)
        M_b = sm.diff(E, theta)

        

        M_b_prev = sm.diff(E, theta_prev)
        M_b_next = sm.diff(E, theta_next)
        #put all expressions in a dictionary to return including intermediate expressions
        expressions = {
            'E': E,
            'M_d': M_d,
            'M_b': M_b,
            'M_b_prev': M_b_prev,
            'M_b_next': M_b_next,
            'beta': beta,
            'gamma': gamma,
            'd0': d0,
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'd4': d4,
            'd5': d5,
            'd6': d6,
            'd6_prev': d6_prev,
            'd6_next': d6_next,
            'K1_tog': K1_tog,
            'K2_tog': K2_tog,
            'K3_tog': K3_tog,
            'C4_tog': C4_tog,
        }

        return expressions
    
    def create_lambda_functions(self):
        #Define the symbolic expression for the dirac delta function so that it evaluates to zero for physical reasons
        dirac_delta_zero = lambda x: 0

        self.num_funcs = {}
        for key, expression in self.expressions.items():
            self.num_funcs[key] = sm.lambdify(
                (self.alpha_sym, self.theta_sym, self.u_sym,self.theta_prev_sym, self.theta_next_sym), expression, {"DiracDelta": dirac_delta_zero})
    
    def update_state(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta

    def update_neighbors_state(self):
        self.left_neighbor_theta = self.left_neighbor.theta if self.left_neighbor is not None else self.params['theta0']
        self.right_neighbor_theta = self.right_neighbor.theta if self.right_neighbor is not None else self.params['theta0']

    def compute_cell_energy(self, u):
        self.update_neighbors_state()
        # Evaluate the lambdified functions
        E = self.num_funcs['E'](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)
        return E

    def compute_force_residuals(self, u):

        self.update_neighbors_state()

        M_d = self.num_funcs['M_d'](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)
        M_b = self.num_funcs['M_b'](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)
        M_b_prev = self.num_funcs['M_b_prev'](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)
        M_b_next = self.num_funcs['M_b_next'](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)

        return M_d, M_b, M_b_prev, M_b_next
    
    def compute_quantities(self, u):
        self.update_neighbors_state()
        quantities = {}
        for key, _ in self.expressions.items():
            quantities[key] = self.num_funcs[key](self.alpha, self.theta, u,self.left_neighbor_theta,self.right_neighbor_theta)
        return quantities

    
    
    def compute_points(self):

        BE_DE_gap = self.params['L4']

        beta = self.num_funcs['beta'](self.alpha, self.theta, 0,self.left_neighbor_theta,self.right_neighbor_theta)
        gamma = self.num_funcs['gamma'](self.alpha, self.theta, 0,self.left_neighbor_theta,self.right_neighbor_theta)


        end_effector = np.array([-self.params['L1']*np.sin(self.alpha), self.params['L1']*np.cos(self.alpha)])
        bistable_shuttle = np.array([0,2*self.params['L1']*np.cos(self.alpha)])
        bistable_anchor = bistable_shuttle + np.array([self.params['L2']*np.cos(beta),self.params['L2']*np.sin(beta)])
        bistable_anchor2 = bistable_shuttle + np.array([-self.params['L2']*np.cos(beta),self.params['L2']*np.sin(beta)])
        signal_router = bistable_shuttle + np.array([self.params['L3']*np.cos(gamma),self.params['L3']*np.sin(gamma)])
        signal_router2 = signal_router + np.array([BE_DE_gap/4,0])
        bifurcation_anchor = np.array([BE_DE_gap, self.params['L1']*np.cos(self.params['alpha0'])+self.params['L4']*np.cos(self.params['theta0'])])
        bifurcation_shuttle = bifurcation_anchor + np.array([-self.params['L4']*np.sin(self.theta),-self.params['L4']*np.cos(self.theta)])
        bifurcation_signal = bifurcation_anchor+np.array([-self.L5*np.sin(self.theta),-self.L5*np.cos(self.theta)])
        bifurcation_input = bifurcation_anchor+np.array([0,-2*self.params['L4']*np.cos(self.theta)])


        points = np.array([
            np.array([0, 0]), #origin
            end_effector, #end effector
            bistable_shuttle, #bistable shuttle
            bistable_anchor, #right bistable anchor
            bistable_anchor2, #left bistable anchor
            signal_router, #signal router
            signal_router2, #signal router 2
            bifurcation_signal, #bifurcation signal
            bifurcation_anchor, #bifurcation anchor
            bifurcation_shuttle, #bifurcation shuttle
            bifurcation_input, #bifurcation input
        ])
        #convert points to dtype float
        points = points.astype(float)
        return points

    def compute_lines(self):
        points = self.compute_points()
        tristable_lines = [
            [points[0], points[1]], #origin to end effector
            [points[1], points[2]], #end effector to bistable shuttle
            [points[2], points[3]], #bistable shuttle to right bistable anchor
            [points[2], points[4]], #bistable shuttle to left bistable anchor
        ]
        signal_router_lines = [
            [points[2], points[5]], #bistable shuttle to signal router
            [points[5], points[6]], #signal router to signal router 2
        ]
        bifurcation_lines = [
            [points[8], points[9]], #bifurcation anchor to bifurcation shuttle
            [points[9], points[10]], #bifurcation shuttle to bifurcation input
        ]
        spring_length = np.linalg.norm(points[7]-points[6])
        signal_spring = Spring(ne=4, a=spring_length, r0=3e-3)
        ss_xs, ss_ys= signal_spring.compute(*points[7], *points[6])
        #compute the lines of the signal spring, 
        signal_spring_lines = [[(ss_xs[i],ss_ys[i]),(ss_xs[i+1],ss_ys[i+1])] for i in range(len(ss_xs)-1)]

        return tristable_lines, signal_router_lines, bifurcation_lines, signal_spring_lines



class System:
    def __init__(self, init_config, params):
        self.n = len(init_config)
        self.u_init = -1e-3
        self.u = self.u_init
        self.cell_spacing = 10e-3
        self.threshold = 1e-3
        self.params = params
        
        # Initialize unit cells
        alpha_init = params['alpha0']
        theta_init = params['theta0']

        self.unit_cells = [UnitCell(alpha_init, theta_init, params,i) for i in range(self.n)]
        self.state_vector = np.zeros(2*self.n)
        # Connect unit cells
        for i,cell in enumerate(self.unit_cells):
            if i > 0:
                cell.left_neighbor = self.unit_cells[i - 1]
            if i < self.n - 1:
                cell.right_neighbor = self.unit_cells[i + 1]

            self.state_vector[2*i] = cell.alpha
            self.state_vector[2*i+1] = cell.theta    


        self.alphaub = np.deg2rad(30)
        self.alphalb = np.deg2rad(-30)

        self.thetaub = np.deg2rad(60)
        self.thetalb = np.deg2rad(-60)

        #define lower and upper bounds by repeating the upper and lower bounds for each unit cell
        self.lb = np.array([self.alphalb,self.thetalb]*self.n)
        self.ub = np.array([self.alphaub,self.thetaub]*self.n)
        self.bounds = opt.Bounds(self.lb,self.ub)

        self.set_configuration(init_config)

    def set_bounds(self, alb, aub, tlb, tub):

        self.lb = np.array([alb, tlb]*self.n)
        self.ub = np.array([aub, tub]*self.n)
        self.bounds.lb = self.lb
        self.bounds.ub = self.ub

    def update_state(self, state_vector):
        """
        Updates the state of the system given the current state vector.

        Args:
        state_vector (numpy.ndarray): The current state vector.
        """
        self.state_vector = state_vector
        for cell in self.unit_cells:
            cell.update_state(state_vector[2*cell.i], state_vector[2*cell.i+1])

    def compute_energy(self, state_vector=None):
        """
        Computes the energy of the system given the current state vector.

        Args:
        state_vector (numpy.ndarray): The current state vector.

        Returns:
        energy (float): The energy of the system.
        """
        # Update state in each cell
        if state_vector is not None:
            self.update_state(state_vector)
        else:
            state_vector = self.state_vector

        # Compute energy
        energy = 0
        for cell in self.unit_cells:
            energy += cell.compute_cell_energy(self.u)

        return energy
    
    def compute_residuals(self, state_vector=None):
        """
        Computes the residuals of the system given the current state vector.

        Args:
        state_vector (numpy.ndarray): The current state vector.

        Returns:
        residuals (numpy.ndarray): The residuals of the system.
        """
        # Update state in each cell
        if state_vector is not None:
            self.update_state(state_vector)
        else:
            state_vector = self.state_vector
        # Compute residuals
        moment_residuals = np.zeros(2*self.n)
        force_residuals = np.zeros(2*self.n)
        for cell in self.unit_cells:
            M_d, M_b,M_b_prev,M_b_next = cell.compute_force_residuals(self.u)
            moment_residuals[2*cell.i] = M_d
            moment_residuals[2*cell.i+1] = M_b

            if cell.left_neighbor is not None:
                moment_residuals[2*cell.i-1] += M_b_prev
            if cell.right_neighbor is not None:
                moment_residuals[2*cell.i+3] += M_b_next
        return moment_residuals
    
    def compute_force_residuals(self, state_vector=None):
        # Update state in each cell
        if state_vector is not None:
            self.update_state(state_vector)
        else:
            state_vector = self.state_vector

        force_residuals = np.zeros(2*self.n)
        for cell in self.unit_cells:
            M_d, M_b,M_b_prev,M_b_next = cell.compute_force_residuals(self.u)
            force_residuals[2*cell.i] = M_d/(self.params['L1']*np.cos(cell.alpha))
            force_residuals[2*cell.i+1] = M_b/(self.params['L4']*np.cos(cell.theta))
            if cell.left_neighbor is not None:
                force_residuals[2*cell.i-1] += M_b_prev/(self.params['L4']*np.cos(cell.left_neighbor.theta))
            if cell.right_neighbor is not None:
                force_residuals[2*cell.i+3] += M_b_next/(self.params['L4']*np.cos(cell.right_neighbor.theta))
        return force_residuals

    def get_quantity(self, quantity, state_vector=None):

        # Update state in each cell
        if state_vector is not None:
            self.update_state(state_vector)
        else:
            state_vector = self.state_vector

        # Compute quantity
        result = np.zeros(self.n)
        for cell in self.unit_cells:
            result[cell.i] = cell.num_funcs[quantity](cell.alpha, cell.theta, self.u,cell.left_neighbor_theta,cell.right_neighbor_theta)

        return result
    
    def get_intermediate_quantities(self, state_vector=None):

        # Update state in each cell
        if state_vector is not None:
            self.update_state(state_vector)
        else:
            state_vector = self.state_vector

        # Compute quantities
        results = {key: [cell.compute_quantities(self.u)[key] for cell in self.unit_cells]
               for key in self.unit_cells[0].expressions.keys()}
        return results

        
    
    
    def solve_equilibrium(self):
        """
        Solves the system of equations to find the equilibrium state of the unit cell.
        This method collects all force residuals and solves the system of equations using a suitable numerical method.
        """
        # Collect all force residuals
        equilibria = opt.minimize(self.compute_energy, self.state_vector,bounds=self.bounds, method='SLSQP', jac=self.compute_residuals, options={'ftol': 1e-12, 'disp': False})


        # Solve the system of equations here using a suitable numerical method
        return equilibria.x

    def simulate_actuation(self, d6max = 33e-3,n_cycles = 1,frames_per_cycle = 1000):

        umax = 2*(self.params['L4']-np.sqrt(self.params['L4']**2-d6max**2))
        t = np.linspace(0, n_cycles, int(frames_per_cycle*n_cycles))
        u_range = (sig.sawtooth(2*np.pi*t, width=0.5)+1)/2*(umax-self.u_init)+self.u_init

        #initialize the states and quantities
        states = np.zeros((len(u_range), 2*self.n))
        quantities = {key: np.ndarray((self.n,len(u_range))) for key in self.unit_cells[0].expressions.keys()}


        for i,u in enumerate(u_range):
            prev_u = self.u
            self.u = u
            
            #resetting mechanism
            if prev_u<self.threshold and u>self.threshold:
                self.set_bounds(self.params['alpha0'],self.params['alpha0'],self.thetalb,self.thetaub)
            else:
                self.set_bounds(self.alphalb,self.alphaub,self.thetalb,self.thetaub)

            self.state_vector = self.solve_equilibrium()
            states[i] = self.state_vector

            current_quantities = self.get_intermediate_quantities()
            for key in quantities.keys():
                quantities[key][:,i] = current_quantities[key]

            

            
        return states, quantities, t
    
    def get_force_response(self, n=250):
        """
        Computes the force response of the system for a given range of angles of the left leg.
        """
        alpha0 = self.params['alpha0']
        alpha_range = np.linspace(alpha0, -alpha0, n)
        force_residuals = np.zeros((len(alpha_range), 2))
        d0 = self.params['L1']*(np.sin(self.params['alpha0'])-np.sin(alpha_range))
        bub = np.copy(self.bounds.ub)
        blb = np.copy(self.bounds.lb)
        original_ub = np.copy(self.bounds.ub)
        original_lb = np.copy(self.bounds.lb)

        for i, alpha in enumerate(alpha_range):
        # update the bounds
            blb[::2] = alpha
            bub[::2] = alpha

            self.bounds.ub = bub
            self.bounds.lb = blb
        # solve for the equilivrium state
            self.solve_equilibrium()

        # get the force residuals
            force_residuals[i] = self.compute_force_residuals()[0]
        force_response = -force_residuals[:, 0]

        #restore the bounds
        self.bounds.ub = original_ub
        self.bounds.lb = original_lb

        return force_response, d0
    
    def get_threshold_stiffnesses(self, d6max = 33e-3):
        force_response, d0 = self.get_force_response()

        results = phf.find_tangent_lines(np.vstack((d0,force_response)).T,[d6max,0])
        return results

    def plot_system(self, fig = None, ax = None, plot_indicators = False):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.axes(projection='3d')

        xlim = [-0.025, 0.1]
        ylim = [-0.05, self.cell_spacing*self.n+0.05]
        zlim = [-0.03, 0.1]
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(0, -90)
        ax.set_proj_type('ortho')
        

        self.tristable_lines = []
        self.tristable_joints = []
        self.signal_router_lines = []
        self.signal_router_joints = []
        self.bifurcation_lines = []
        self.bifurcation_joints = []
        self.signal_spring_lines = []
        self.self_spring_lines = []
        self.left_spring_lines = []
        self.right_spring_lines = []
        self.indicators = []
        
        for cell in self.unit_cells:
            points = cell.compute_points()
            tristable_lines, signal_router_lines, bifurcation_lines, signal_spring_lines = cell.compute_lines()

            z_loc = cell.i*self.cell_spacing
            tristable_mech = ax.add_collection3d(LineCollection(tristable_lines,color='k'), zs=z_loc, zdir='y')
            self.tristable_lines.append(tristable_mech)
            signal_router_mech = ax.add_collection3d(LineCollection(signal_router_lines,color='k'), zs=z_loc, zdir='y')
            self.signal_router_lines.append(signal_router_mech)
            bifurcation_mech = ax.add_collection3d(LineCollection(bifurcation_lines,color='k'), zs=z_loc, zdir='y')
            self.bifurcation_lines.append(bifurcation_mech)
            signal_spring_mech = ax.add_collection3d(LineCollection(signal_spring_lines,color='k'), zs=z_loc, zdir='y')
            self.signal_spring_lines.append(signal_spring_mech)

            z_locs = np.array([z_loc]*points.shape[0])
            tristable_joints = ax.scatter(points[:5, 0], z_locs[:5], points[:5, 1], c='w', marker='o',depthshade=False,edgecolors='k')
            self.tristable_joints.append(tristable_joints)
            signal_router_joints = ax.scatter(points[5:8, 0], z_locs[5:8], points[5:8, 1], c='w', marker='o',depthshade=False,edgecolors='k')
            self.signal_router_joints.append(signal_router_joints)
            bifurcation_joints = ax.scatter(points[8:, 0], z_locs[8:], points[8:, 1], c='w', marker='o',depthshade=False,edgecolors='k')
            self.bifurcation_joints.append(bifurcation_joints)
            
            
            cell.self_spring = Spring(ne=10, a=0.05, r0=3e-3)
            cell.self_spring.initialize_leads(*points[9],z_loc, *points[1],z_loc)
            ss_xs, ss_zs, ss_ys = cell.self_spring.compute_with_leads(*points[9],z_loc, *points[1],z_loc)
            self_spring_lines = [[(ss_xs[i],ss_ys[i],ss_zs[i]),(ss_xs[i+1],ss_ys[i+1],ss_zs[i+1])] for i in range(len(ss_xs)-1)]
            self_spring_mech = ax.add_collection3d(Line3DCollection(self_spring_lines,color='g'))
            self.self_spring_lines.append(self_spring_mech)

            if cell.left_neighbor is not None:
                cell.left_spring = Spring(ne=10, a=0.05, r0=3e-3)
                left_points = cell.left_neighbor.compute_points()
                cell.left_spring.initialize_leads(*points[1],z_loc, *left_points[9],z_loc-self.cell_spacing)
                ss_xs, ss_zs, ss_ys = cell.left_spring.compute_with_leads(*points[1],z_loc, *left_points[9],z_loc-self.cell_spacing)
                left_spring_lines = [[(ss_xs[i],ss_ys[i],ss_zs[i]),(ss_xs[i+1],ss_ys[i+1],ss_zs[i+1])] for i in range(len(ss_xs)-1)]
                left_spring_mech = ax.add_collection3d(Line3DCollection(left_spring_lines,color='b'))
                self.left_spring_lines.append(left_spring_mech)
            else:
                self.left_spring_lines.append(None)
            if cell.right_neighbor is not None:
                cell.right_spring = Spring(ne=10, a=0.05, r0=3e-3)
                right_points = cell.right_neighbor.compute_points()
                cell.right_spring.initialize_leads(*points[1],z_loc, *right_points[9],z_loc+self.cell_spacing)
                ss_xs, ss_zs, ss_ys = cell.right_spring.compute_with_leads(*points[1],z_loc, *right_points[9],z_loc+self.cell_spacing)
                right_spring_lines = [[(ss_xs[i],ss_ys[i],ss_zs[i]),(ss_xs[i+1],ss_ys[i+1],ss_zs[i+1])] for i in range(len(ss_xs)-1)]
                right_spring_mech = ax.add_collection3d(Line3DCollection(right_spring_lines,color='r'))
                self.right_spring_lines.append(right_spring_mech)
            else:
                self.right_spring_lines.append(None)
            
            #plot a point above the bifurcation element with a box marker
            indicator = ax.scatter(points[7,0], z_loc, points[7,1]+30e-3, marker='s', s=100, edgecolors='k', zorder=10,color='k')
            self.indicators.append(indicator)
            # color the box marker white if the bifurcation element is in the off state, which is when theta is positive
            if cell.theta > 0:
                plt.setp(indicator, color='w')
            if not plot_indicators:
                #hide the indicator
                plt.setp(indicator, visible=False)

        ax.set_xlim3d(-0.025, 0.1)
        ax.set_ylim3d(-0.05, self.cell_spacing*self.n+0.05)
        ax.set_zlim3d(-0.03, 0.1)
        ax.set_position((0,0,1,1))

        return fig, ax

    def update_plot(self, frame, states):
        self.update_state(states[frame])

        def update_lines(lines):
            return [[(x[0][0], z_loc, x[0][1]), (x[1][0], z_loc, x[1][1])]
            for x in lines
            ]


        for i, cell in enumerate(self.unit_cells):
            z_loc = cell.i * self.cell_spacing
            points = cell.compute_points()
            tristable_lines, signal_router_lines, bifurcation_lines, signal_spring_lines = cell.compute_lines()
            tristable_lines = update_lines(tristable_lines)
            signal_router_lines = update_lines(signal_router_lines)
            bifurcation_lines = update_lines(bifurcation_lines)
            signal_spring_lines = update_lines(signal_spring_lines)
            
            # Update tristable mechanism lines
            self.tristable_lines[i].set_segments(tristable_lines)

            # Update signal router mechanism lines
            self.signal_router_lines[i].set_segments(signal_router_lines)

            # Update bifurcation mechanism lines
            self.bifurcation_lines[i].set_segments(bifurcation_lines)

            # Update signal spring mechanism lines
            self.signal_spring_lines[i].set_segments(signal_spring_lines)

            # Update joints (scatter points)
            # Update tristable joints
            self.tristable_joints[i]._offsets3d = (points[:5, 0], z_loc * np.ones(5), points[:5, 1])

            # Update signal router joints
            self.signal_router_joints[i]._offsets3d = (points[5:8, 0], z_loc * np.ones(3), points[5:8, 1])

            # Update bifurcation joints
            self.bifurcation_joints[i]._offsets3d = (points[8:, 0], z_loc * np.ones(points.shape[0]-8), points[8:, 1])
            # Update self spring mechanism

            ss_xs, ss_zs, ss_ys = cell.self_spring.compute_with_leads(*points[9], z_loc, *points[1], z_loc)
            self_spring_lines = [[(ss_xs[j], ss_ys[j], ss_zs[j]), (ss_xs[j+1], ss_ys[j+1], ss_zs[j+1])] for j in range(len(ss_xs)-1)]
            self.self_spring_lines[i].set_segments(self_spring_lines)

            # Update left spring mechanism
            if cell.left_neighbor is not None:
                left_points = cell.left_neighbor.compute_points()
                ls_xs, ls_zs, ls_ys = cell.left_spring.compute_with_leads(*points[1], z_loc, *left_points[9], z_loc-self.cell_spacing)
                left_spring_lines = [[(ls_xs[j], ls_ys[j], ls_zs[j]), (ls_xs[j+1], ls_ys[j+1], ls_zs[j+1])] for j in range(len(ls_xs)-1)]
                self.left_spring_lines[i].set_segments(left_spring_lines)

            # Update right spring mechanism
            if cell.right_neighbor is not None:
                right_points = cell.right_neighbor.compute_points()
                rs_xs, rs_zs, rs_ys = cell.right_spring.compute_with_leads(*points[1], z_loc, *right_points[9], z_loc+self.cell_spacing)
                right_spring_lines = [[(rs_xs[j], rs_ys[j], rs_zs[j]), (rs_xs[j+1], rs_ys[j+1], rs_zs[j+1])] for j in range(len(rs_xs)-1)]
                self.right_spring_lines[i].set_segments(right_spring_lines)

            # Update indicator colour
            if cell.theta>0:
                plt.setp(self.indicators[i], color='w')
            else:
                plt.setp(self.indicators[i], color='k')
            
        return self.tristable_lines + self.signal_router_lines + self.bifurcation_lines + self.signal_spring_lines + self.tristable_joints + self.signal_router_joints + self.bifurcation_joints + self.self_spring_lines + self.left_spring_lines + self.right_spring_lines + self.indicators


    
    def get_configuration(self):
        thetas = [cell.theta for cell in self.unit_cells]
        
        return [1 if theta<0 else 0 for theta in thetas]
    
    def set_configuration(self, configuration):
        #check that configuration is valid
        assert len(configuration) == self.n, "Configuration must be a list of length n"
        assert all([c==0 or c==1 for c in configuration]), "Configuration must be a list of 0s and 1s"

        # print("Setting configuration to {}".format(configuration))
        for i in range(self.n):
            self.state_vector[2*i] = 0 if configuration[i] else self.params['alpha0']
        self.solve_equilibrium()

class Spring:
    """
    A class representing a spring.

    Attributes:
    -----------
    ne : int
        Number of elements in the spring.
    a : float
        Length of the spring.
    r0 : float
        Natural radius of the spring.
    Li_2 : float
        Square of the length of each element of the spring.
    ei : numpy.ndarray
        vector of longitudinal coordinates of the spring elements.
    b : numpy.ndarray
        vector of transverse coordinates of the spring elements.
    """

    def __init__(self, ne=None, a=None, r0=None):
        self.ne = ne
        self.a = a
        self.r0 = r0
        self.Li_2 = None
        self.ei = None
        self.b = None
        if all([ne, a, r0]):
            self._initialize()

    def _initialize(self):
        """
        Initializes the spring parameters.
        """
        self.Li_2 = (self.a / (4 * self.ne))**2 + self.r0**2
        self.ei = np.arange(2 * self.ne + 2)
        j = np.arange(2 * self.ne)
        self.b = np.concatenate(([0], (-1)**j, [0]))

    def compute(self, xa, ya, xb, yb):
        """
        Computes the position of the spring.

        Parameters:
        -----------
        xa : float
            x-coordinate of the starting point of the spring.
        ya : float
            y-coordinate of the starting point of the spring.
        xb : float
            x-coordinate of the ending point of the spring.
        yb : float
            y-coordinate of the ending point of the spring.

        Returns:
        --------
        xs : numpy.ndarray
            Array of x-coordinates of the spring elements.
        ys : numpy.ndarray
            Array of y-coordinates of the spring elements.
        """
        if self.ne is None or self.a is None or self.r0 is None:
            raise ValueError("Spring parameters not initialized!")
        
        R = np.array([xb, yb]) - np.array([xa, ya])
        mod_R = np.linalg.norm(R)
        L_2 = (mod_R / (4 * self.ne))**2
        
        if L_2 > self.Li_2:
            raise ValueError("Initial conditions cause pulling the spring beyond its maximum large. Try reducing these conditions.")
        else:
            r = np.sqrt(self.Li_2 - L_2)
        
        c = r * self.b
        u1 = R / mod_R
        u2 = np.array([-u1[1], u1[0]])
        
        xs = xa + u1[0] * (mod_R / (2 * self.ne + 1)) * self.ei + u2[0] * c
        ys = ya + u1[1] * (mod_R / (2 * self.ne + 1)) * self.ei + u2[1] * c
        
        return xs, ys
    
    def compute3d(self, xa, ya, za, xb, yb, zb):
        if self.ne is None or self.a is None or self.r0 is None:
            raise ValueError("Spring parameters not initialized!")
        
        R = np.array([xb, yb, zb]) - np.array([xa, ya, za])
        mod_R = np.linalg.norm(R)
        L_2 = (mod_R / (4 * self.ne))**2
        
        if L_2 > self.Li_2:
            raise ValueError("Initial conditions cause pulling the spring beyond its maximum length. Try reducing these conditions.")
        else:
            r = np.sqrt(self.Li_2 - L_2)
        
        u1 = R / mod_R
        # Define the yv vector perpendicular to the xz-plane
        yv = np.array([0,1,0])


        u3 = np.cross(yv, u1)
        # Compute the u3 vector which is the cross product of yv and u2
        u2 = np.cross(u1, u3)

        
        xs = xa + u1[0] * (mod_R / (2 * self.ne + 1)) * self.ei + u2[0] * r * self.b
        ys = ya + u1[1] * (mod_R / (2 * self.ne + 1)) * self.ei + u2[1] * r * self.b
        zs = za + u1[2] * (mod_R / (2 * self.ne + 1)) * self.ei + u2[2] * r * self.b
        
        return xs, ys, zs
    
    def initialize_leads(self, xa, ya, za, xb, yb, zb):
        # Calculate the total distance between points A and B
        R = np.array([xb, yb, zb]) - np.array([xa, ya, za])
        mod_R = np.linalg.norm(R)
        
        # Calculate lead length (buffer on each side)
        self.lead_length = (mod_R - self.a) / 2
        
        # Calculate the direction vector between points A and B
        self.u1_lead = R / mod_R

    def compute_with_leads(self, xa, ya, za, xb, yb, zb):
        R = np.array([xb, yb, zb]) - np.array([xa, ya, za])
        mod_R = np.linalg.norm(R)
        self.u1_lead = R / mod_R
        # Calculate the new start and end points for the spring using lead lengths
        spring_start = np.array([xa, ya, za]) + self.u1_lead * self.lead_length
        spring_end = np.array([xb, yb, zb]) - self.u1_lead * self.lead_length
        
        # Get spring points using the 3D compute method
        xs, ys, zs = self.compute3d(spring_start[0], spring_start[1], spring_start[2],
                                  spring_end[0], spring_end[1], spring_end[2])
        
        # Add the lead sections to the spring points
        xs = np.concatenate(([xa], xs, [xb]))
        ys = np.concatenate(([ya], ys, [yb]))
        zs = np.concatenate(([za], zs, [zb]))

        return xs, ys, zs