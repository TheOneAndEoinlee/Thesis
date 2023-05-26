from ansys.mapdl.core import launch_mapdl
import numpy as np
import os 

def generate_flexure_keypoints(mapdl, flexure_length, reinforcement_length,theta,initial_keypoint=None):
    if initial_keypoint is None:
        initial_keypoint = mapdl.k("", 0, 0, 0)
    
    origin = mapdl.geometry.keypoints[initial_keypoint-1]

    k1 = np.array([flexure_length, 0, 0])+origin
    k2 = np.array([reinforcement_length*np.cos(theta), reinforcement_length*np.sin(theta), 0])+k1
    k3 = np.array([flexure_length, 0, 0])+k2

    keypoints = [
        initial_keypoint,
        mapdl.k("",*k1),
        mapdl.k("",*k2),
        mapdl.k("",*k3)
    ]

    return keypoints
def generate_flexure_lines(mapdl, keypoints):
    lines = [
        mapdl.l(keypoints[0], keypoints[1]),
        mapdl.l(keypoints[1], keypoints[2]),
        mapdl.l(keypoints[2], keypoints[3])
    ]

    return lines

def mesh_flexure(mapdl, lines, beam_thickness, beam_depth, flexure_thickness):
    # Create a rectangular area section for the reinforced beam
    mapdl.sectype(1, "BEAM", "RECT", 1)
    mapdl.secdata(beam_thickness, beam_depth,2,2)

    #mesh the reinforced beam
    mapdl.lmesh(lines[1])



    # Create a rectangular area section for the flexure
    mapdl.sectype(2, "BEAM", "RECT", 1)
    mapdl.secdata(flexure_thickness, beam_depth,2,2)

    #mesh the flexure
    mapdl.lmesh(lines[0])
    mapdl.lmesh(lines[2])


def main():
    # Initialize the MAPDL session
    path = os.getcwd()
    jname = 'basic_beam'
    mapdl = launch_mapdl(run_location=path, jobname=jname, override=True)

    # Define parameters
    beam_length = 1e-3
    flexure_length = 0.1e-3
    beam_thickness = 1e-4
    flexure_thickness = 0.2e-4
    beam_depth = 1e-4

    # Define material properties and element type
    mapdl.prep7()
    mapdl.et(1, "BEAM188")
    mapdl.mp("EX", 1, 200e9)  # Young's modulus
    mapdl.mp("PRXY", 1, 0.3)  # Poisson's ratio

    # Define keypoints and create a line between them
    link1 = generate_flexure_keypoints(mapdl, flexure_length, beam_length, np.pi/8)

    # Plot the keypoints and lines

    mapdl.kplot(
    show_keypoint_numbering=True,
    background="black",
    show_bounds=True,
    font_size=26,
)
    lines = generate_flexure_lines(mapdl, link1)

    #mesh the flexure
    mesh_flexure(mapdl, lines, beam_thickness, beam_depth, flexure_thickness)
    
    

    

    # mapdl.eplot(
    # background="black",
    # show_bounds=True,
    # font_size=26,
    # show_element_numbering=True,
    # )

    
    # Exit the MAPDL session
    mapdl.finish()
    mapdl.exit()

if __name__ == "__main__":
    main()