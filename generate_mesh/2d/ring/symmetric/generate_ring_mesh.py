'''
This code generates a  ring  mesh with radial symmetry: symmetry is obtained by replicating a ring slice
The inner ring is tagged with tag 'circle_r_id', the outer ring is tagged with tag 'circle_R_id', and all radial lines (spokes) are tagged with 'radial_lines_id'

Run it with
    python3 generate_ring_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/ring/symmetric"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/ring/symmetric/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_ring_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
import math
import meshio
import sys
import numpy as np

module_path = '/home/fenics/shared/modules'

sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

M = int(np.round(math.log2(rpam.parameters["N"])))
theta = 2 * np.pi / rpam.parameters["N"]

output_directory = io.add_trailing_slash(rarg.args.output_directory)
mesh_slice_file = output_directory + "ring_slice/mesh.msh"
mesh_xdmf_file = output_directory + "mesh.xdmf"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


# generate the ring slice and save it to mesh_slice_file
msh.generate_mesh_ring_slice(rpam.parameters["r"], rpam.parameters["R"], rpam.parameters["c_r"], rpam.parameters["c_R"], theta, rpam.parameters["resolution"], mesh_slice_file)

# Load the mesh slice
mesh = meshio.read(mesh_slice_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')

# initialize the loop over 0 <= theta < 2 pi by setting the initial values of the extremal points of the first ring slice
r_1 = np.array([rpam.parameters["r"], 0])
r_2 = cal.R(theta).dot(r_1)
r_4 = np.array([rpam.parameters["R"], 0])
r_3 = cal.R(theta).dot(r_4)

print('Looping through circle ...')

for i in range(1, M + 1):
    # at each step of this loop, a slice is doubled in size by mirroring, until a full ring is constructed

    # print(f'\t i = {i}')

    # set the extremal points of the new ring slice in terms of the old ones
    r_1 = np.copy(r_2)
    r_2 = cal.R(2 ** (i - 1) * theta).dot(r_1)
    r_4 = np.copy(r_3)
    r_3 = cal.R(2 ** (i - 1) * theta).dot(r_4)

    # define the axis of symmetry according to the current mirroring operation
    gamma_axis_of_symmetry = lambda t: cal.line(r_1, r_4, t)

    '''
    # define the function which tells whetehr a point lies on the current axis of symmetry
    def point_on_axis_of_symmetry(point):
        return cal.point_on_line(point, gamma_axis_of_symmetry)

    # define the function which makes current mirroring operation
    def mirror_function(point):
        return cal.mirror_point_line(point, gamma_axis_of_symmetry)


    # Mirror points across gamma_top
    old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
                                                                                                       mesh.point_data)
    msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
    msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
    '''
    msh.mirror_mesh(mesh, gamma_axis_of_symmetry)

# tag circle_r: extract the lines whose starting point is part of  circle_r by considering its distance with respect to the circle center
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(np.linalg.norm(np.subtract(mesh.points[line[0]], rpam.parameters["c_r"])), rpam.parameters["r"]) and np.isclose(np.linalg.norm(np.subtract(mesh.points[line[1]], rpam.parameters["c_r"])), rpam.parameters["r"])),
    rpam.parameters["circle_r_id"], mesh
)

# tag circle_R: extract the lines whose starting point is part of  circle_R by considering its distance with respect to the circle center
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(np.linalg.norm(np.subtract(mesh.points[line[0]], rpam.parameters["c_R"])), rpam.parameters["R"]) and np.isclose(np.linalg.norm(np.subtract(mesh.points[line[1]], rpam.parameters["c_R"])), rpam.parameters["R"])),
    rpam.parameters["circle_R_id"], mesh
)

# rag the radial lines
msh.asssign_tag_to_lines(lambda line: cal.line_is_radial(line, rpam.parameters["N"], mesh), rpam.parameters["radial_lines_id"], mesh)

print('... done.')
meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

# msh.print_mesh_info(mesh, 'Mesh after mirroring')

# read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh_from_file = meshio.read(mesh_xdmf_file)

# print line mesh
line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_directory + "line_mesh.xdmf", line_mesh)

# print triangle mesh
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)

# print  mesh vertices
mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

# print mesh metadata
io.write_parameters_to_csv_file(mesh_metadata_file_name, metadata)
