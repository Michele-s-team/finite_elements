'''
generate a mesh given by a square circle with a square inside

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/inner_square"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/inner_square/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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

surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

# add a 0d object:


# gmsh.model.mesh.embed(1, [line_34], 2, surface)
# gmsh.model.geo.synchronize()

# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)

gmsh.model.addPhysicalGroup(vertices[0][0], [vertices[0][1]], rpam.parameters["p_1_id"])
gmsh.model.setPhysicalName(vertices[0][0], rpam.parameters["p_1_id"], "p_1")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_12_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_12_id"], "line_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_23_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_23_id"], "line_23")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_34_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_34_id"], "line_34")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_41_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_41_id"], "line_41")



# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["surface_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["surface_id"], "surface")

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution_min"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution_max"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", rpam.parameters["r_resolution_min"])
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["r_resolution_max"])

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], rpam.parameters, output_directory, True)

model.__exit__()
