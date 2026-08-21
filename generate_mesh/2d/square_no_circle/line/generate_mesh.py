'''
generate a mesh given by a square whose top line is a one-dimensional submesh

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
    
Here 'sub_mesh_0' is the two-dimensional square mesh and 'sub_mesh_1' is the one-dimensional top edge of the square.
'''

from fenics import *
import gmsh
import math
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_directory = io.add_trailing_slash(rarg.args.output_directory)
sub_mesh_1_output_directory = io.add_trailing_slash(output_directory + 'sub_mesh_1')
mesh_file = output_directory + "mesh.msh"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'

print("output_directory = ", output_directory)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add outer rectangle
p_1 = gmsh.model.geo.addPoint(0, 0, 0)
p_2 = gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0)
p_3 = gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0)
p_4 = gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)
gmsh.model.geo.synchronize()

line_12 = gmsh.model.geo.addLine(p_1, p_2)
line_23 = gmsh.model.geo.addLine(p_2, p_3)
line_34 = gmsh.model.geo.addLine(p_3, p_4)
line_41 = gmsh.model.geo.addLine(p_4, p_1)
gmsh.model.geo.synchronize()

loop = gmsh.model.geo.addCurveLoop([line_12, line_23, line_34, line_41])
gmsh.model.geo.synchronize()

surface_square = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# DEBUG: Print what line entities we have
print(f"DEBUG: Line entities: {lines}")
print(f"DEBUG: Expected line IDs - line_12: {line_12}, line_23: {line_23}, line_34: {line_34}, line_41: {line_41}")

# square lines
gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_sub_mesh_0_b_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_sub_mesh_0_b_id"], "line_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_sub_mesh_0_r_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_sub_mesh_0_r_id"], "line_23")

# gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_sub_mesh_0_t_id"])
# gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_sub_mesh_0_t_id"], "line_34")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["sub_mesh_1_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["sub_mesh_1_id"], "sub_mesh_1")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_sub_mesh_0_l_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_sub_mesh_0_l_id"], "line_41")

# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

# DEBUG: Print what surface entities we have
print(f"DEBUG: Surface entities: {surfaces}")

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["sub_mesh_0_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["sub_mesh_0_id"], "sub_mesh_0")

# DEBUG: Print the physical group IDs we're using
print(f"DEBUG: Physical group IDs:")
print(f"  sub_mesh_0_id (surface): {rpam.parameters['sub_mesh_0_id']}")
print(f"  sub_mesh_1_id (line_34): {rpam.parameters['sub_mesh_1_id']}")

# set the resolution close to the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [loop])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(rpam.parameters["L"], rpam.parameters["h"]))

gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

# print the boundary points of the boundaries given by the top line (sub_mesh 1)
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    [rpam.parameters['sub_mesh_1_id']],
    os.path.join(output_directory, 'boundary_points_id_' + str(rpam.parameters['sub_mesh_1_id']) + '.csv'))



msh.clear_gmsh()

# ========================================================================
# Generate submesh for the top edge from the 2D mesh and save it in .h5 format
# ========================================================================

print("Generating H5 sub_mesh for top edge from 2D mesh...")

# Read the generated 2D mesh from the triangle component file
mesh_temp = Mesh()
with XDMFFile(output_directory + "triangle_mesh.xdmf") as infile:
    infile.read(mesh_temp)


# create a list of the vertices in mesh_2d which lie on the top edge
top_edge_vertices = []
for vertex in vertices(mesh_temp):
    point = vertex.point()
    if math.isclose(point.y(), rpam.parameters["h"]):
        top_edge_vertices.append(point.x())

# Sort vertices by x-coordinate and remove duplicates
top_edge_vertices = sorted(list(set(top_edge_vertices)))

print(f"Found {len(top_edge_vertices)} unique vertices on top edge")

# Create a proper 1D IntervalMesh using the actual vertex positions
if len(top_edge_vertices) >= 2:

    num_intervals = len(top_edge_vertices) - 1

    # Create output directory for submesh
    sub_mesh_1_output_directory = output_directory + "sub_meshes/1/"
    os.makedirs(sub_mesh_1_output_directory, exist_ok=True)

    sub_mesh_1_metadata = dict([])
    sub_mesh_1_metadata['x_l'] = 0.0
    sub_mesh_1_metadata['x_r'] = rpam.parameters['L']
    sub_mesh_1_metadata['coordinates'] = top_edge_vertices
    sub_mesh_1_metadata['resolution'] = rpam.parameters['resolution']
    sub_mesh_1_metadata['line_id'] = rpam.parameters['sub_mesh_1_id']
    sub_mesh_1_metadata['vertex_l_id'] = rpam.parameters['vertex_sub_mesh_1_l_id']
    sub_mesh_1_metadata['vertex_r_id'] = rpam.parameters['vertex_sub_mesh_1_r_id']
    sub_mesh_1_metadata['file_format'] = 'h5'

    # generate the line mesh with the specific coordinates written in top_edge_vertices, which may not be equally spaced
    msh.genereate_line_mesh(0.0, rpam.parameters['L'], num_intervals,
                            rpam.parameters['sub_mesh_1_id'], rpam.parameters['vertex_sub_mesh_1_l_id'], rpam.parameters['vertex_sub_mesh_1_r_id'],
                            output_directory=sub_mesh_1_output_directory, metadata=sub_mesh_1_metadata,
                            coordinates=top_edge_vertices)


    print("...done!")
    
else:
    print("Error: Not enough vertices found on top edge")
