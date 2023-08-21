import sympy as sm


class Bistable:
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
    def __init__(self, d1_init, params):
        self.d1 = d1_init
        self.params = params  

        self.define_symbols_and_parameters()

        self.expressions = self.define_symbolic_functions()
        
        self.create_lambda_functions()


    def define_symbols_and_parameters(self):
        self.d1_sym= sm.symbols('d1')
        self.params_symbols = {
            'beta0': sm.symbols('beta0'),
            'L2': sm.symbols('L2'),
            'C1': sm.symbols('C1'),
            'Kf2': sm.symbols('Kf2'),

        }

    def define_symbolic_functions(self):
        
        # Extract symbols
        d1 = self.d1_sym
        params = self.params_symbols.copy()
        params.update(self.params) # Merge with numerical parameters

        
        beta0 = params['beta0']
        L2 = params['L2']
        C1 = params['C1']
        Kf2 = params['Kf2']
        beta = sm.asin((L2*sm.sin(beta0)-d1)/L2) #angle of bistable shuttle with respect to the horizontal axis
        d2 = (L2*sm.sin(beta0)-d1)/(sm.tan(beta))-L2*sm.cos(beta0) #horizontal displacement of the bistable support
        

        # define the energy of the system
        E = 0.5*(8*Kf2*(beta-beta0)**2
                 + C1*(d2)**2
                 )

        # define the force of the system
        #This is a bug, this is the moment of the force, since alpha is the angle of the left leg
        #this explains the other bug!!!
        f_d = sm.diff(E, d1)

        
        #put all expressions in a dictionary to return including intermediate expressions
        expressions = {
            'E': E,
            'beta': beta,
            'd1': d1,
            'd2': d2,
            'f_d': f_d
        }

        return expressions
    
    def create_lambda_functions(self):

        self.num_funcs = {}
        for key, expression in self.expressions.items():
            self.num_funcs[key] = sm.lambdify(
                (self.d1_sym), expression)
    
    def update_state(self, d1):
        self.d1 = d1

    def compute_cell_energy(self):

        # Evaluate the lambdified functions
        E = self.num_funcs['E'](self.d1)
        return E

    def compute_force_residuals(self):


        fd = self.num_funcs['f_d'](self.d1)
        return fd
