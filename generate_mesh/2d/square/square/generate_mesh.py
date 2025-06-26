'''
generate a mesh given by a square circle with a square inside

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/square"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/square/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
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
gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_out_b_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_out_b_id"], "line_out_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_out_r_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_out_r_id"], "line_out_23")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_out_t_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_out_t_id"], "line_out_34")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_out_l_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_out_l_id"], "line_out_41")


# inner lines
gmsh.model.addPhysicalGroup(lines[4][0], [lines[4][1]], rpam.parameters["line_in_b_id"])
gmsh.model.setPhysicalName(lines[4][0], rpam.parameters["line_in_b_id"], "line_in_12")

gmsh.model.addPhysicalGroup(lines[5][0], [lines[5][1]], rpam.parameters["line_in_r_id"])
gmsh.model.setPhysicalName(lines[5][0], rpam.parameters["line_in_r_id"], "line_in_23")

gmsh.model.addPhysicalGroup(lines[6][0], [lines[6][1]], rpam.parameters["line_in_t_id"])
gmsh.model.setPhysicalName(lines[6][0], rpam.parameters["line_in_t_id"], "line_in_34")

gmsh.model.addPhysicalGroup(lines[7][0], [lines[7][1]], rpam.parameters["line_in_l_id"])
gmsh.model.setPhysicalName(lines[7][0], rpam.parameters["line_in_l_id"], "line_in_41")



# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["surface_out_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["surface_out_id"], "surface_out")

gmsh.model.addPhysicalGroup(surfaces[1][0], [surfaces[1][1]], rpam.parameters["surface_in_id"])
gmsh.model.setPhysicalName(surfaces[1][0], rpam.parameters["surface_in_id"], "surface_in")


# set the resolution
# se resolution resolution_min at distance r_resolution_min from surface_in, and resolution_amx at distance r_resolution_max from surface_out_id
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface_in])

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

msh.full_write(mesh_file, ['triangle', 'line'], rpam.parameters, output_directory, True)


#####
# create the submesh and its functions to read triangles and lines
parent_mesh = msh.read_mesh(output_directory + 'triangle_mesh.xdmf')
sf = msh.read_mesh_components(parent_mesh, parent_mesh.topology().dim(), output_directory + "triangle_mesh.xdmf")
mf = msh.read_mesh_components(parent_mesh, parent_mesh.topology().dim()-1, output_directory + "line_mesh.xdmf")

submesh_out = SubMesh(parent_mesh, sf, rpam.parameters["surface_out_id"])

sf_submesh_out = msh.transfer_cell_tags_to_submesh(submesh_out, sf)
mf_submesh_out = msh.transfer_facet_tags_to_submesh(parent_mesh, submesh_out, mf)

with XDMFFile(output_directory + "triangle_sub_mesh.xdmf") as xdmf:
    xdmf.write(submesh_out)
    xdmf.write(sf_submesh_out)

submesh_out_boundary = BoundaryMesh(submesh_out, "exterior", order=True)
with XDMFFile(output_directory + "line_sub_mesh.xdmf") as xdmf:
    xdmf.write(submesh_out_boundary)

#####

model.__exit__()
