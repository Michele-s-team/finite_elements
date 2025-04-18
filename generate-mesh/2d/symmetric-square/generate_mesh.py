'''
This Code generates a symmetric square mesh.
If you want to generate the mesh form the terminal use :
    clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.01 $SOLUTION_PATH


    where resolution is the mesh size and output_dir is the directory where to save the mesh
    The half mesh will be saved in the output_dir as mesh.msh, while the complete mesh as mesh.xdmf
    The mesh will be saved in the output_dir as line_mesh.xdmf and triangle_mesh.xdmf
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
import list as lis
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_dir")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
r = 0.3
L = 1
h = 1
c_r = [L / 2, h / 2, 0]

output_dir = args.output_dir
half_mesh_msh_file = output_dir + "/half_mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

'''
This function duplicates and transform all points, inverting their position with respect to the x axis
- 'points' : Array of points to be duplicated
- 'point_data' : Data that contains dimensional tag of the points (must be duplicated as well to avoid issues during the reading of the mesh)
Outputs
- 'new_points' : the old and the new points
- 'non_mirrored_new_points_indices' : the indices of the old points which have not been mirrored, and of the 
newly mirrored points in the new array 
(they are not just the indices of the old points traslated by some constant since the points on the x axis has not been duplicated and they were not ordered in the old list)
- 'mirrored_point_data ': array of the points which have been mirrored 
'''


def mirror_points(points, point_data):
    offset = 0
    non_mirrored_plus_new_points_indices = []
    mirrored_points = []
    mirrored_point_data = []

    print('Called mirror_points')

    # lis.print_list(points, 'old points')

    print('Looping through points...')

    for i in range(len(points)):
        if np.isclose(points[i, 1], 0, rtol=1e-3):
            # I ran into a point with x[1] = 0 -> do not mirror it and append to old_plus_new_points the same index 'i' as the original point
            offset += 1
            non_mirrored_plus_new_points_indices.append(i)

            # print(f'\tNot mirroring points with label {i}')

        else:
            #  I ran into a point with x[1] != 0 -> mirror it
            non_mirrored_plus_new_points_indices.append(i - offset + len(points))
            l = list(point_data['gmsh:dim_tags'][i, :])

            # append two points with indexes:
            # 1) the original point
            mirrored_point_data.append(l)
            # 2) the mirror of hte original point
            mirrored_points.append([points[i, 0], h - points[i, 1], points[i, 2]])

            # print(f'\tMirroring points with label {i}')

    print('... done.')

    mirrored_points = np.array(mirrored_points)
    old_plus_new_points = np.vstack((points, mirrored_points))

    # lis.print_list(old_plus_new_points, 'old + new points')
    # lis.print_list(non_mirrored_plus_new_points_indices, 'non-mirrored + new points indices')
    # lis.print_list(mirrored_point_data, 'mirrored point data')

    return old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data


'''
Half mesh is generated used pygmsh and it's saved as mesh.msh
'''
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(np.pi / resolution))
'''
construct a rectangle with vertices [L,h/2], [L,h], [0,h], [0,h/2]
'''
half_rectangle_points = [model.add_point((L, h / 2, 0), mesh_size=resolution * (min(L, h) / r)),
                         model.add_point((L, h, 0), mesh_size=resolution * (min(L, h) / r)),
                         model.add_point((0, h, 0), mesh_size=resolution * (min(L, h) / r)),
                         model.add_point((0, h / 2, 0), mesh_size=resolution * (min(L, h) / r)),
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
l_edge_id = 1
r_edge_id = 2
t_edge_id = 3
b_edge_id = 4
circle_id = 5
ids = [1, np.nan, r_edge_id, l_edge_id, t_edge_id, circle_id]
# Load the half-mesh
mesh = meshio.read(half_mesh_msh_file)
print("original points", np.shape(mesh.points))

# Mirror points across X=0
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = mirror_points(mesh.points,
                                                                                               mesh.point_data)

old_triangles = mesh.cells_dict['triangle']
# lis.print_list(old_triangles, 'original triangles')

original_lines = mesh.cells_dict['line']
# lis.print_list(original_lines, 'original lines')


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
# lis.print_list(new_triangles, 'new triangles')


mesh.points = old_plus_new_points
mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], mirrored_point_data))
mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((old_triangles, new_triangles)))
# print(mesh.cells[-1])
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



# assign to the l edge the id 'lower_edge_id'
for j in range(len(mesh.cells)):
    # loop through  blocks of lines

    if mesh.cells[j].type == "line":
        # print(f'\tI am on line block {mesh.cells[j].data}')

        # loop through the lines in  block  mesh.cells[j].data
        for i in range(len(mesh.cells[j].data)):

            # obtain the extremal point of each line
            point1 = mesh.points[mesh.cells[j].data[i][0]]
            point2 = mesh.points[mesh.cells[j].data[i][1]]

            if (np.isclose(point1[1], 0, rtol=1e-3) and np.isclose(point2[1], 0, rtol=1e-3)):
                # the extremal points lie on the axis x[1] = 0 -> the line mesh.cells[j].data[i] belongs to the b edge of the rectangle
                # print(f"\t\tLine: {i} -> Point 1: {point1}, Point 2: {point2}")
                # tag the line under consideration with ID target_id
                mesh.cell_data['gmsh:physical'][j][i] = b_edge_id

meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")

'''
This part read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
'''

mesh_from_file = meshio.read(mesh_xdmf_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

print("Mesh generated and saved to ", output_dir)
