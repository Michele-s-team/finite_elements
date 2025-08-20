'''
generate a mesh given by a half circle with a line inside

Run it with
    python3 generate_half_circle_with_line_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/1d/line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

# mesh resolution
output_directory = io.add_trailing_slash(rarg.args.output_directory)
mesh_file = output_directory + "mesh.msh"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

print("output_directory = ", output_directory)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 0d object:
point_l = gmsh.model.geo.addPoint(rpam.parameters['x_l'], 0, 0)
point_r = gmsh.model.geo.addPoint(rpam.parameters['x_r'], 0, 0)
gmsh.model.geo.synchronize()

line = gmsh.model.geo.addLine(point_l, point_r)
gmsh.model.geo.synchronize()

# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)

gmsh.model.addPhysicalGroup(vertices[0][0], [vertices[0][1]], rpam.parameters["point_l_id"])
gmsh.model.setPhysicalName(vertices[0][0], rpam.parameters["point_l_id"], "point_l")

gmsh.model.addPhysicalGroup(vertices[1][0], [vertices[1][1]], rpam.parameters["point_r_id"])
gmsh.model.setPhysicalName(vertices[1][0], rpam.parameters["point_r_id"], "point_r")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_id"], "line")

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [line])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["x_r"]-rpam.parameters["x_l"])

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=1)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['line', 'vertex'], rpam.parameters, output_directory, True)

model.__exit__()

