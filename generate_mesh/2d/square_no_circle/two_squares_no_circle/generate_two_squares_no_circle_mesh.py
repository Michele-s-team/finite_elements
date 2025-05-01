'''
generate a  mesh given by two collated squares

run it with
python3 generate_two_squares_no_circle_mesh.py [resolution] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_two_squares_no_circle_mesh.py 0.1 $SOLUTION_PATH

'''

import meshio
import gmsh
import pygmsh
import argparse
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)

# add '/' to output_directory if it is missing
output_directory = args.output_directory
output_directory = io.add_trailing_slash(output_directory)

mesh_file = output_directory + "mesh.msh"





# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

# CHANGE PARAMETERS HERE
# L and h are length and height of the rectangle
L = 1
h = 2
# L_m is the coordinate on the x axis where the inner line separating the two sub-rectangles is located
L_m = L / 3
l_surface_id = 1
r_surface_id = 2
l_line_id = 3
lb_line_id = 4
rb_line_id = 5
r_line_id = 6
tr_line_id = 7
tl_line_id = 8
m_line_id = 9
# CHANGE PARAMETERS HERE

print("L = ", L)
print("h = ", h)
print("resolution = ", resolution)


# Create corner points
p_lb = gmsh.model.geo.addPoint(0, 0, 0)
p_mb = gmsh.model.geo.addPoint(L_m, 0, 0)
p_mt = gmsh.model.geo.addPoint(L_m, h, 0)
p_lt = gmsh.model.geo.addPoint(0, h, 0)

p_rb = gmsh.model.geo.addPoint(L, 0, 0)
p_rt = gmsh.model.geo.addPoint(L, h, 0)

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
gmsh.model.addPhysicalGroup(2, [surface_l], l_surface_id)
gmsh.model.setPhysicalName(2, l_surface_id, "left_square")

gmsh.model.addPhysicalGroup(2, [surface_r], r_surface_id)
gmsh.model.setPhysicalName(2, r_surface_id, "right_square")

# tag lines
gmsh.model.addPhysicalGroup(1, [l_lt_lb], l_line_id)
gmsh.model.setPhysicalName(1, l_line_id, "l_line")

gmsh.model.addPhysicalGroup(1, [l_lb_mb], lb_line_id)
gmsh.model.setPhysicalName(1, lb_line_id, "lb_line")

gmsh.model.addPhysicalGroup(1, [l_rb_mb], rb_line_id)
gmsh.model.setPhysicalName(1, rb_line_id, "rb_line")

gmsh.model.addPhysicalGroup(1, [l_rt_rb], r_line_id)
gmsh.model.setPhysicalName(1, r_line_id, "r_line")

gmsh.model.addPhysicalGroup(1, [l_mt_rt], tr_line_id)
gmsh.model.setPhysicalName(1, tr_line_id, "tr_line")

gmsh.model.addPhysicalGroup(1, [l_mt_lt], tl_line_id)
gmsh.model.setPhysicalName(1, tl_line_id, "tl_line")

gmsh.model.addPhysicalGroup(1, [l_mb_mt], m_line_id)
gmsh.model.setPhysicalName(1, m_line_id, "m_line")

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface_l])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * L)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", L)

circle_r_dist = gmsh.model.mesh.field.add("Distance")
circle_r_threshold = gmsh.model.mesh.field.add("Threshold")

gmsh.model.mesh.field.setNumber(circle_r_threshold, "IField", circle_r_dist)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, circle_r_threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

# Mesh and write
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

gmsh.finalize()

mesh_from_file = meshio.read(mesh_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_directory + "line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)
