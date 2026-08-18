'''
generate a mesh given by a half circle with a line inside

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/half_circle_with_line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/half_circle_with_line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
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

# mesh resolution
output_directory = io.add_trailing_slash(rarg.args.output_directory)
mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


c_1 = [rpam.parameters["r"], 0, 0]
c_2 = [-rpam.parameters["r"], 0, 0]

print("output_directory = ", output_directory)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 0d object:
p_1 = gmsh.model.geo.addPoint(c_1[0], c_1[1], c_1[2])
p_2 = gmsh.model.geo.addPoint(c_2[0], c_2[1], c_2[2])
p_c = gmsh.model.geo.addPoint(0, 0, 0)
p_3 = gmsh.model.geo.addPoint(rpam.parameters["c_3"][0], rpam.parameters["c_3"][1], rpam.parameters["c_3"][2])
p_4 = gmsh.model.geo.addPoint(rpam.parameters["c_4"][0], rpam.parameters["c_4"][1], rpam.parameters["c_4"][2])
gmsh.model.geo.synchronize()

line_12 = gmsh.model.geo.addLine(p_1, p_2)
gmsh.model.geo.synchronize()

arc_21 = gmsh.model.geo.addCircleArc(p_2, p_c, p_1)
gmsh.model.geo.synchronize()

line_34 = gmsh.model.geo.addLine(p_3, p_4)
gmsh.model.geo.synchronize()

loop = gmsh.model.geo.addCurveLoop([line_12, arc_21])
gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, [line_34], 2, surface)
gmsh.model.geo.synchronize()

# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)

gmsh.model.addPhysicalGroup(vertices[0][0], [vertices[0][1]], rpam.parameters["p_1_id"])
gmsh.model.setPhysicalName(vertices[0][0], rpam.parameters["p_1_id"], "p_1")

gmsh.model.addPhysicalGroup(vertices[1][0], [vertices[1][1]], rpam.parameters["p_2_id"])
gmsh.model.setPhysicalName(vertices[1][0], rpam.parameters["p_2_id"], "p_2")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_12_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_12_id"], "line_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["arc_21_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["arc_21_id"], "arc_12")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_34_id"])
gmsh.model.setPhysicalName(lines[2][0], line_34, "line_34")

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

msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], metadata, output_directory, True)

msh.clear_gmsh(geometry)
