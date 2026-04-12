'''
generate a  mesh given by two collated squares

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/two_squares_no_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/two_squares_no_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'



# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()


# Create corner points
p_lb = gmsh.model.geo.addPoint(0, 0, 0)
p_mb = gmsh.model.geo.addPoint(rpam.parameters["L_m"], 0, 0)
p_mt = gmsh.model.geo.addPoint(rpam.parameters["L_m"], rpam.parameters["h"], 0)
p_lt = gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)

p_rb = gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0)
p_rt = gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0)

# Left square lines and surface
l_lb_mb = gmsh.model.geo.addLine(p_lb, p_mb)
l_mb_mt = gmsh.model.geo.addLine(p_mb, p_mt)
l_mt_lt = gmsh.model.geo.addLine(p_mt, p_lt)
l_lt_lb = gmsh.model.geo.addLine(p_lt, p_lb)
loop_l = gmsh.model.geo.addCurveLoop([l_lb_mb, l_mb_mt, l_mt_lt, l_lt_lb])
surface_l = gmsh.model.geo.addPlaneSurface([loop_l])

# Right square lines and surface
l_mt_rt = gmsh.model.geo.addLine(p_mt, p_rt)
l_rt_rb = gmsh.model.geo.addLine(p_rt, p_rb)
l_rb_mb = gmsh.model.geo.addLine(p_rb, p_mb)
loop_r = gmsh.model.geo.addCurveLoop([l_mt_rt, l_rt_rb, l_rb_mb, l_mb_mt])
surface_r = gmsh.model.geo.addPlaneSurface([loop_r])

# tag objects
# Synchronize and tag surfaces
gmsh.model.geo.synchronize()

msh.tag_physical_object((2, surface_l), rpam.parameters['l_surface_id'], gmsh.model, 'left_square')
msh.tag_physical_object((2, surface_r), rpam.parameters['r_surface_id'], gmsh.model, 'right_square')


# tag lines
# gmsh.model.addPhysicalGroup(1, [l_lt_lb], rpam.parameters['l_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['l_line_id'], "l_line")

msh.tag_physical_object((1, l_lt_lb), rpam.parameters['l_line_id'], gmsh.model, 'l_line')


# gmsh.model.addPhysicalGroup(1, [l_lb_mb], rpam.parameters['lb_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['lb_line_id'], "lb_line")

msh.tag_physical_object((1, l_lb_mb), rpam.parameters['lb_line_id'], gmsh.model, 'lb_line')


# gmsh.model.addPhysicalGroup(1, [l_rb_mb], rpam.parameters['rb_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['rb_line_id'], "rb_line")

msh.tag_physical_object((1, l_rb_mb), rpam.parameters['rb_line_id'], gmsh.model, 'rb_line')


# gmsh.model.addPhysicalGroup(1, [l_rt_rb], rpam.parameters['r_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['r_line_id'], "r_line")

msh.tag_physical_object((1, l_rt_rb), rpam.parameters['r_line_id'], gmsh.model, 'r_line')


# gmsh.model.addPhysicalGroup(1, [l_mt_rt], rpam.parameters['tr_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['tr_line_id'], "tr_line")

msh.tag_physical_object((1, l_mt_rt), rpam.parameters['tr_line_id'], gmsh.model, 'tr_line')


# gmsh.model.addPhysicalGroup(1, [l_mt_lt], rpam.parameters['tl_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['tl_line_id'], "tl_line")

msh.tag_physical_object((1, l_mt_lt), rpam.parameters['tl_line_id'], gmsh.model, 'tl_line')


# gmsh.model.addPhysicalGroup(1, [l_mb_mt], rpam.parameters['m_line_id'])
# gmsh.model.setPhysicalName(1, rpam.parameters['m_line_id'], "m_line")

msh.tag_physical_object((1, l_mb_mt), rpam.parameters['m_line_id'], gmsh.model, 'm_line')


# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface_l])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * rpam.parameters["L"])
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["L"])

circle_r_dist = gmsh.model.mesh.field.add("Distance")
circle_r_threshold = gmsh.model.mesh.field.add("Threshold")

gmsh.model.mesh.field.setNumber(circle_r_threshold, "IField", circle_r_dist)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, circle_r_threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

# Mesh and write
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

gmsh.finalize()
