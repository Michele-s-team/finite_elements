'''
generate a mesh given by a square circle with a square inside

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/square"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/square/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
import gmsh
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
mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'



print("output_directory = ", output_directory)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add outer rectangle
p_out_1 = gmsh.model.geo.addPoint(0, 0, 0)
p_out_2 = gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0)
p_out_3 = gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0)
p_out_4 = gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)
gmsh.model.geo.synchronize()

line_out_12 = gmsh.model.geo.addLine(p_out_1, p_out_2)
line_out_23 = gmsh.model.geo.addLine(p_out_2, p_out_3)
line_out_34 = gmsh.model.geo.addLine(p_out_3, p_out_4)
line_out_41 = gmsh.model.geo.addLine(p_out_4, p_out_1)
gmsh.model.geo.synchronize()

loop_out = gmsh.model.geo.addCurveLoop([line_out_12, line_out_23, line_out_34, line_out_41])
gmsh.model.geo.synchronize()

# add inner rectangle
p_in_1 = gmsh.model.geo.addPoint(rpam.parameters["p"][0], rpam.parameters["p"][1], rpam.parameters["p"][2])
p_in_2 = gmsh.model.geo.addPoint(rpam.parameters["p"][0] + rpam.parameters["L_in"], rpam.parameters["p"][1], rpam.parameters["p"][2])
p_in_3 = gmsh.model.geo.addPoint(rpam.parameters["p"][0] + rpam.parameters["L_in"], rpam.parameters["p"][1] + rpam.parameters["h_in"], rpam.parameters["p"][2])
p_in_4 = gmsh.model.geo.addPoint(rpam.parameters["p"][0], rpam.parameters["p"][1] + rpam.parameters["h_in"], rpam.parameters["p"][2])
gmsh.model.geo.synchronize()

line_in_12 = gmsh.model.geo.addLine(p_in_1, p_in_2)
line_in_23 = gmsh.model.geo.addLine(p_in_2, p_in_3)
line_in_34 = gmsh.model.geo.addLine(p_in_3, p_in_4)
line_in_41 = gmsh.model.geo.addLine(p_in_4, p_in_1)
gmsh.model.geo.synchronize()

loop_in = gmsh.model.geo.addCurveLoop([line_in_12, line_in_23, line_in_34, line_in_41])
gmsh.model.geo.synchronize()

surface_out = gmsh.model.geo.addPlaneSurface([loop_out, loop_in])
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, [line_in_12, line_in_23, line_in_34, line_in_41], 2, surface_out)
gmsh.model.geo.synchronize()

surface_in = gmsh.model.geo.addPlaneSurface([loop_in])
gmsh.model.geo.synchronize()

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# outer lines
gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_sub_mesh_1_b_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_sub_mesh_1_b_id"], "line_out_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_sub_mesh_1_r_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_sub_mesh_1_r_id"], "line_out_23")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_sub_mesh_1_t_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_sub_mesh_1_t_id"], "line_out_34")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_sub_mesh_1_l_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_sub_mesh_1_l_id"], "line_out_41")

# inner lines
gmsh.model.addPhysicalGroup(lines[4][0], [lines[4][1]], rpam.parameters["line_sub_mesh_0_b_id"])
gmsh.model.setPhysicalName(lines[4][0], rpam.parameters["line_sub_mesh_0_b_id"], "line_in_12")

gmsh.model.addPhysicalGroup(lines[5][0], [lines[5][1]], rpam.parameters["line_sub_mesh_0_r_id"])
gmsh.model.setPhysicalName(lines[5][0], rpam.parameters["line_sub_mesh_0_r_id"], "line_in_23")

gmsh.model.addPhysicalGroup(lines[6][0], [lines[6][1]], rpam.parameters["line_sub_mesh_0_t_id"])
gmsh.model.setPhysicalName(lines[6][0], rpam.parameters["line_sub_mesh_0_t_id"], "line_in_34")

gmsh.model.addPhysicalGroup(lines[7][0], [lines[7][1]], rpam.parameters["line_sub_mesh_0_l_id"])
gmsh.model.setPhysicalName(lines[7][0], rpam.parameters["line_sub_mesh_0_l_id"], "line_in_41")

# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["sub_mesh_1_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["sub_mesh_1_id"], "surface_out")

gmsh.model.addPhysicalGroup(surfaces[1][0], [surfaces[1][1]], rpam.parameters["sub_mesh_0_id"])
gmsh.model.setPhysicalName(surfaces[1][0], rpam.parameters["sub_mesh_0_id"], "surface_in")

# set the resolution
# se resolution equal to parameters["resolution"] at buth distance 0 from surface_in, and  at distance max(rpam.parameters["L"],rpam.parameters["h"]) from sub_mesh_1_id
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface_in])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(rpam.parameters["L"], rpam.parameters["h"]))

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

msh.generate_sub_mesh(output_directory, os.path.join(output_directory, 'sub_meshes', 'in'), rpam.parameters["sub_mesh_0_id"])
msh.generate_sub_mesh(output_directory, os.path.join(output_directory, 'sub_meshes', 'out'), rpam.parameters["sub_mesh_1_id"])

msh.clear_gmsh()
