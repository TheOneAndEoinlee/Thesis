import numpy as np
from ansys.mapdl.core import launch_mapdl

def generate_keypoints(mapdl, start_keypoints, end_keypoints, flexure_length):
    keypoints = []

    if start_keypoints is None:
        start_keypoints = [
            mapdl.k("", 0, 0, 0),
            mapdl.k("", flexure_length, 0, 0),
        ]

    if end_keypoints is None:
        last_keypoint_x = mapdl.queries.kx(start_keypoints[-1])
        end_keypoints = [
            mapdl.k("", last_keypoint_x + flexure_length, 0, 0),
            mapdl.k("", last_keypoint_x + 2 * flexure_length, 0, 0),
        ]

    keypoints.extend(start_keypoints)
    keypoints.extend(end_keypoints)

    return keypoints



def generate_beam_elements(mapdl, keypoints, central_thickness, flexure_thickness):
    # Define material properties and element type
    mapdl.prep7()
    mapdl.et(1, "BEAM188")
    mapdl.mp("EX", 1, 200e9)  # Young's modulus
    mapdl.mp("PRXY", 1, 0.3)  # Poisson's ratio

    # Create flexure segments
    mapdl.sectype(1, "BEAM", "RECT", 1)
    mapdl.secdata(flexure_thickness, flexure_thickness)
    mapdl.e(keypoints[0], keypoints[1])
    mapdl.e(keypoints[2], keypoints[3])

    # Create central reinforced sections
    mapdl.sectype(2, "BEAM", "RECT", 1)
    mapdl.secdata(central_thickness, central_thickness)
    mapdl.e(keypoints[1], keypoints[2])



def create_mechanism(mapdl, central_thickness, flexure_thickness, flexure_length, start_keypoints=None, end_keypoints=None):
    keypoints = generate_keypoints(mapdl, start_keypoints, end_keypoints, flexure_length)
    generate_beam_elements(mapdl, keypoints, central_thickness, flexure_thickness)
    return keypoints

def apply_boundary_conditions(mapdl):
    # Apply boundary conditions (fix, force, etc.) as required
    # Example:
    mapdl.d(1, "UX", 0)
    mapdl.d(1, "UY", 0)
    mapdl.d(1, "UZ", 0)
    mapdl.f(2, "FX", 1000)  # Apply a force of 1000 N in the x-direction


def run_analysis(mapdl):
    # Run the analysis and extract results
    mapdl.run("/SOLU")
    mapdl.solve()
    mapdl.finish()

    mapdl.post1()
    mapdl.set(1, 1)
    displacement = mapdl.etable("DISP", "LS", 2, "UZ")
    return displacement

import matplotlib.pyplot as plt

def plot_results(nodal_displacement):
    x_disp = nodal_displacement[:, 0]
    y_disp = nodal_displacement[:, 1]
    z_disp = nodal_displacement[:, 2]

    # Create a plot using the displacement data
    plt.plot(x_disp, label="X Displacement")
    plt.plot(y_disp, label="Y Displacement")
    plt.plot(z_disp, label="Z Displacement")
    plt.xlabel("Node Number")
    plt.ylabel("Displacement")
    plt.legend()
    plt.show()

def main():
    mapdl = launch_mapdl()
    mapdl.prep7()
    # Input parameters
    central_thickness = 0.5
    flexure_thickness = 0.1
    flexure_length = 2

    # Create first mechanism
    first_mechanism_keypoints = create_mechanism(mapdl, central_thickness, flexure_thickness, flexure_length)

    # Create second mechanism connected to the first mechanism
    second_mechanism_keypoints = create_mechanism(
        mapdl,
        central_thickness,
        flexure_thickness,
        flexure_length,
        start_keypoints=[first_mechanism_keypoints[2], first_mechanism_keypoints[3]],
    )

    # Apply boundary conditions, run the analysis, and plot the results
    apply_boundary_conditions(mapdl)
    nodal_displacement = run_analysis(mapdl)
    plot_results(nodal_displacement)
    mapdl.exit()

if __name__ == "__main__":
    main()
