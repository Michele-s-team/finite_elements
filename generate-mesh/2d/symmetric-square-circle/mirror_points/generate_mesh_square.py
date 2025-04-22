'''
This code generates a symmetric square mesh with a circular hole in it.
Symmetry is enforced by mirroring the mesh points along a symetry axis.

run with
python3 generate_mesh_square.py [mesh resolution] [path where to store the mesh]
ATTENTION: [mesh resolution] must be small enough for the circle to be properly resolved
Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_square.py 0.3 $SOLUTION_PATH

The half mesh will be saved in [path where to store the mesh] as half_mesh.msh. The complete mesh will be saved in
[path where to store the mesh] as mesh.xdmf, triangle_mesh.xdmf, line_mesh.xdmf and vertices.csv.
'''

import meshio
from fenics import *
import gmsh  # main tool
import pygmsh  # wrapper for gmsh
import argparse
import sys
import numpy as np

# add the path where to find the shared modules
# gaetano's path
# module_path = '/home/tanos/Thesis/finite_elements/modules/'
# michele's path
module_path = '/home/fenics/shared/modules'

sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_dir")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
r = 0.25
L = 2
h = 1
y_coordinate_axis_of_symmetry = h / 2
c_r = [L / 2, y_coordinate_axis_of_symmetry, 0]

'''
this function tells whether a point lies on the axis of symmetry
Input values:
- 'coordinate' : the coordinates of the point (list of two values)
Return value:
- True/False, if the point lies on the axis of symmetry 
'''
def point_on_axis_of_symmetry(point):
    gamma_axis_of_symmetry = lambda t: cal.line([0, h / 2], [L, h / 2], t)
    cal.point_on_line(point, gamma_axis_of_symmetry)


output_dir = args.output_dir
half_mesh_msh_file = output_dir + "/half_mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

print(f'L = {L}\nh = {h}\nc_r = {c_r}\nresolution = {resolution}\noutput directory = {output_dir}')


'''
Half mesh is generated used pygmsh and it's saved as mesh.msh
'''
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(r * np.pi / resolution))

'''
construct a rectangle with vertices [L,h/2], [L,h], [0,h], [0,h/2]
'''
half_rectangle_points = [model.add_point((L, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
                         model.add_point((L, h, 0), mesh_size=resolution),
                         model.add_point((0, h, 0), mesh_size=resolution),
                         model.add_point((0, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
                         ]
model.synchronize()

half_circle_points = [
    model.add_point((c_r[0] + -r * np.cos(np.pi * i / N), c_r[1] + r * np.sin(np.pi * i / N), 0), mesh_size=resolution)
    for i in range(N + 1)]
model.synchronize()

half_rectangle_circle_points = half_rectangle_points + half_circle_points
half_rectangle_circle_lines = [model.add_line(half_rectangle_circle_points[i], half_rectangle_circle_points[i + 1])
                               for i in range(-1, len(half_rectangle_circle_points) - 1)]

half_rectangle_circle_loop = model.add_curve_loop(half_rectangle_circle_lines)
half_rectangle_circle_surface = model.add_plane_surface(half_rectangle_circle_loop)

model.synchronize()

model.add_physical([half_rectangle_circle_surface], "Volume")
model.add_physical([half_rectangle_circle_lines[1]], "r")
model.add_physical([half_rectangle_circle_lines[3]], "l")
model.add_physical([half_rectangle_circle_lines[2]], "t")
# model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
model.add_physical(half_rectangle_circle_lines[5:], "c")

geometry.generate_mesh(dim=2)
gmsh.write(half_mesh_msh_file)

# msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()

'''
duplicate the points and cells with the respective tags and ids
The new mesh inherits the ids (physical id used for measure definiton) of the original one,
except for the new physical objects that are generated from reflection (e.g. the b line)

In particular the rule 4:5 implies that the lines that in the original mesh where
in the physical group 4 (top lines), when reflected, they will be assigned the id 5 (used to define measure in the bottom line)

Here the lines are tagged as follows:
- volume: id = 1
- b edge: id = 4: now it is set to np.nan is because the l edge generated here, in the half mesh, will be immaterial when the mesh will be mirrored ->
  a proper ID will be assigned to it later
- r edge: id = 2
- t edge: id = 3
- l edge: id = 1
- circle: id = 5
'''
surface_id = 1
l_edge_id = 2
r_edge_id = 3
t_edge_id = 4
b_edge_id = 5
circle_id = 6
ids = [1, np.nan, r_edge_id, l_edge_id, t_edge_id, circle_id]
# Load the half-mesh
mesh = meshio.read(half_mesh_msh_file)

'''
print('********** Mesh before mirroring: **********')
msh.print_mesh_element_types(mesh)
msh.print_mesh_triangles(mesh)
msh.print_mesh_vertices(mesh)
'''

################################################## mirror the mesh ##################################################



# Mirror points across X=0
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, h, mesh.points,
                                                                                                   mesh.point_data)

old_triangles = mesh.cells_dict['triangle']
old_lines = mesh.cells_dict['line']

# duplicate cell blocks of type 'triangle'
new_triangles = np.copy(old_triangles)
# run through the old triangles
for i in range(np.shape(new_triangles)[0]):
    # for each old triangle, run through each of its three vertices
    for j in range(3):
        '''
        assign to the new triangle the vertex tag of the old triangle, mapped towards the vertex tags of the mirrored vertices
        In this way, one reconstructs the same pattern as the old triangles, for the flipped part of the mesh
        '''
        new_triangles[i, j] = non_mirrored_plus_new_points_indices[old_triangles[i, j]]

mesh.points = old_plus_new_points
mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], mirrored_point_data))
mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((old_triangles, new_triangles)))
N = np.shape(mesh.cells[-1].data)[0]
mesh.cell_data['gmsh:physical'][-1] = np.array([mesh.cell_data['gmsh:physical'][-1][0]] * N)
mesh.cell_data['gmsh:geometrical'][-1] = np.array([mesh.cell_data['gmsh:geometrical'][-1][0]] * N)

# duplicate cell blocks of type 'line'
for j in range(len(mesh.cells)):
    if mesh.cells[j].type == 'line':
        lines = np.copy(mesh.cells[j].data)
        filtered_lines = []
        for i in range(np.shape(lines)[0]):
            f = [mesh.points[lines[i, k]][1] != 0 for k in range(2)]
            if f[0] or f[1]:
                filtered_lines.append([non_mirrored_plus_new_points_indices[lines[i, 0]],
                                       non_mirrored_plus_new_points_indices[lines[i, 1]]])
        filtered_lines = np.array(filtered_lines)
        mesh.cells[j] = meshio.CellBlock("line", np.vstack((lines, filtered_lines)))
        N = np.shape(mesh.cells[j].data)[0]
        mesh.cell_data['gmsh:physical'][j] = np.array([ids[mesh.cell_data['gmsh:physical'][j][0]]] * N)
        mesh.cell_data['gmsh:geometrical'][j] = np.array([mesh.cell_data['gmsh:geometrical'][j][0]] * N)

msh.asssign_tag_to_lines(
    lambda p_start, p_end: (np.isclose(p_start[1], 0, rtol=cal.small_number) and np.isclose(p_end[1], 0, rtol=1e-3)),
    b_edge_id, mesh
)

meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")

'''
print('********** Mesh after mirroring: **********')
msh.print_mesh_element_types(mesh)
msh.print_mesh_triangles(mesh)
msh.print_mesh_vertices(mesh)
'''

# read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh_from_file = meshio.read(mesh_xdmf_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh( output_dir + "/triangle_mesh.xdmf" )
io.print_vertices_to_csv_file( mesh, output_dir + "/vertices.csv" )
