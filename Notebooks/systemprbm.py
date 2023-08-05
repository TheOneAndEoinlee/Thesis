import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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
        bifurcation_anchor = np.array([BE_DE_gap, self.params['L1']*np.cos(self.params['alpha0'])+self.params['L4']*np.cos(self.params['theta0'])])
        bifurcation_shuttle = bifurcation_anchor + np.array([-self.params['L4']*np.sin(self.theta),-self.params['L4']*np.cos(self.theta)])
        bifurcation_signal = bifurcation_anchor+np.array([-self.L5*np.sin(self.theta),-self.L5*np.cos(self.theta)])
        bifurcation_input = bifurcation_anchor+np.array([0,-2*self.params['L4']*np.cos(self.theta)])


        points = np.array([
            np.array([0, 0]), #origin
            end_effector, #end effector
            bistable_shuttle, #bistable shuttle
            bistable_anchor, #right bistable anchor
            bistable_shuttle, #bistable shuttle
            bistable_anchor2, #left bistable anchor
            bistable_shuttle, #bistable shuttle
            signal_router, #signal router
            bifurcation_signal, #bifurcation signal
            bifurcation_anchor, #bifurcation anchor
            bifurcation_shuttle, #bifurcation shuttle
            bifurcation_input, #bifurcation input
        ])
        return points



class System:
    def __init__(self, init_config, params):
        self.n = len(init_config)
        self.u = -1e-3
        self.cell_spacing = 5e-3
        self.params = params
        
        # Initialize unit cells
        alpha_init = 22*np.pi/180
        theta_init = 2*np.pi/180

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
            force_residuals = np.zeros(2*self.n)
            for cell in self.unit_cells:
                F_d, F_b,F_b_prev,F_b_next = cell.compute_force_residuals(self.u)
                force_residuals[2*cell.i] = F_d
                force_residuals[2*cell.i+1] = F_b
                
                #check physical validity of this bit
                if cell.left_neighbor is not None:
                    force_residuals[2*cell.i-1] += F_b_prev
                if cell.right_neighbor is not None:
                    force_residuals[2*cell.i+3] += F_b_next
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
        results = {key: np.array([cell.compute_quantities(self.u)[key] for cell in self.unit_cells])
               for key in self.unit_cells[0].expressions.keys()}
        return results

        
    
    
    def solve_equilibrium(self):
        """
        Solves the system of equations to find the equilibrium state of the unit cell.
        This method collects all force residuals and solves the system of equations using a suitable numerical method.
        """
        # Collect all force residuals
        equilibria = opt.minimize(self.compute_energy, self.state_vector,bounds=self.bounds, method='SLSQP', jac=self.compute_residuals, options={'ftol': 1e-12, 'disp': False})
        self.update_state(equilibria.x)

        # Solve the system of equations here using a suitable numerical method
        return equilibria.x

    def simulate_actuation(self, u_range):
        states = np.zeros((len(u_range), 2*self.n))

        quantities = {key: [] for key in self.unit_cells[0].expressions.keys()}

        for u in u_range:
            self.u = u
            #fix alpha to alpha0 for 2e-3<u<4e-3
            if 1e-3<u<1.5e-3:
                self.set_bounds(self.params['alpha0'],self.params['alpha0'],self.thetalb,self.thetaub)
            else:
                self.set_bounds(self.alphalb,self.alphaub,self.thetalb,self.thetaub)

            self.state_vector = self.solve_equilibrium()
            states[u_range==u] = self.state_vector

            current_quantities = self.get_intermediate_quantities()
            for key in quantities.keys():

                quantities[key].append(current_quantities[key])

            for key in quantities.keys():
                quantities[key] = np.array(quantities[key])

            
        return states, quantities

    def plot_system(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.mechs = []
        for cell in self.unit_cells:
            points = cell.compute_points()
            z_loc = cell.i*self.cell_spacing
            h = plt.plot(points[:, 0],z_loc*np.ones(points.shape[0]), points[:, 1],'k-o')
            self.mechs.append(h)
        ax.set_box_aspect((1,1,1))

        #remove axes
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        #set view
        ax.view_init(0, -87)

        return fig, ax

    def update_plot(self, frame, states):
        self.update_state(states[frame])
        for i,cell in enumerate(self.unit_cells):
            points = cell.compute_points()
            z_loc = cell.i*self.cell_spacing
            self.mechs[i][0].set_data(points[:, 0], z_loc*np.ones(points.shape[0]))
            self.mechs[i][0].set_3d_properties(points[:, 1])
        return self.mechs
    
    def get_configuration(self):
        betas = [cell.beta for cell in self.unit_cells]
        
        return [1 if beta<0 else 0 for beta in betas]
    
    def set_configuration(self, configuration):
        #check that configuration is valid
        assert len(configuration) == self.n, "Configuration must be a list of length n"
        assert all([c==0 or c==1 for c in configuration]), "Configuration must be a list of 0s and 1s"

        for i,cell in enumerate(self.unit_cells):
            cell.alpha = 0 if configuration[i] else np.deg2rad(self.params['alpha0'])
        self.solve_equilibrium()
