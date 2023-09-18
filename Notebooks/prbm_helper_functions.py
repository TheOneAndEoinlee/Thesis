import scipy.signal as sig
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import numpy as np

def find_tangent_lines(data, target_point):
    x_data, y_data = data[:, 0], data[:, 1]

    # Interpolate the data using a spline
    spline = UnivariateSpline(x_data, y_data, s=0)
    spline_derivative = spline.derivative()
    
    def objective(t):
        y_tangent = spline_derivative(t) * (x_data - t) + spline(t)
        y_at_target = spline_derivative(t) * (target_point[0] - t) + spline(t)
        distance = abs(y_at_target - target_point[1])
        return distance

    # Detect peaks for initial guesses
    peaks, _ = sig.find_peaks(y_data)
    peak_x_values = x_data[peaks]

    results = []
    for initial_guess in peak_x_values:
        result = minimize(objective, initial_guess, bounds=[(initial_guess, 1.1*initial_guess)], constraints={'type': 'ineq', 'fun': lambda t: spline(t)})
        tangent_point = result.x[0]
        tangent_value = spline(tangent_point)
        slope = (target_point[1]-tangent_value)/(target_point[0]-tangent_point)
        results.append((tangent_point, float(tangent_value), float(slope)))
    
    # results[0] = (peak_x_values[0], spline(peak_x_values[0]), -spline(peak_x_values[0])/(target_point[0]-peak_x_values[0]))
    return results

def define_params():
    # Define mechanism parameters
    E = 2.2e9 #Young's modulus
    t = 0.4e-3 #thickness
    b = 5e-3    #width
    Lf1 = 6e-3 #length of the flexure on end effector, and bifurcation
    Lf2 = 4e-3 #length of the flexure on signal routing and bistable beams
    Lr1 = 35e-3 #length of the rigid link on end effector
    Lr2 = 24e-3 #length of the rigid link on bistable beams
    Lr3 = 35e-3 #length of the rigid link on signal routing
    Lr4 = 60e-3 #length of the rigid link on bifurcation
    Ls1 = 6e-3 #length of the support on bistable beams
    ts1 = 2e-3 #thickness of the support on bistable beams
    Lsignal = 6e-3 #length of the signal routing spring leg

    alpha0 = 24*np.pi/180 #initial angle of the end effector
    beta0 = 8*np.pi/180 #initial angle of the bistable beams
    gamma0 = 30*np.pi/180 #initial angle of the signal routing
    theta0 = 1*np.pi/180 #initial angle of the bifurcation

    kf1 = E*b*t**3/12/Lf1
    kf2 = E*b*t**3/12/Lf2
    kf3 = E*b*t**3/12/Lf2
    kf4 = E*b*t**3/12/Lf1

    C1 = E*b*ts1**3/Ls1**3 #bistable support stiffness
    C2 = 2*E*b*t**3/Lr2**3 #signal routing guiding stiffness
    C3 = 0.2*E*b*t**3/Lsignal**3/6 #signal routing stiffness
    C4 = 500000 #bifurcation support stiffness

    #todo fix these values

    K1 = 0
    K2 = 0
    K3 = 0

    tog_offset = 3e-3

    params = {
        'L1': Lr1+Lf1,'L2': Lr2+Lf2,'L3': Lr3+Lf2,'L4': Lr4+Lf1,
        'C1': C1,'C2': C2,'C3': C3,'C4': C4,
        'Kf1': kf1,'Kf2': kf2,'Kf3': kf3,'Kf4': kf4,
        'alpha0': alpha0,'beta0': beta0,'gamma0': gamma0,'theta0': theta0,
        'K1': K1,'K2': K2,'K3': K3,
        'tog_offset': tog_offset,
        } 
    return params
