'''
generate a mesh given by a square with a an ellipse embedded in the mesh inside, and a circular hole in the ellipse

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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
mesh_metadata = rpam.parameters.copy()
mesh_metadata['file_format'] = 'xdmf'


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


# add ellipse
p_ellipse_c = gmsh.model.geo.addPoint(rpam.parameters["c"][0], rpam.parameters["c"][1], rpam.parameters["c"][2])
p_ellipse_r = gmsh.model.geo.addPoint(rpam.parameters["c"][0] + rpam.parameters["a"], rpam.parameters["c"][1], rpam.parameters["c"][2])
p_ellipse_t = gmsh.model.geo.addPoint(rpam.parameters["c"][0], rpam.parameters["c"][1] + rpam.parameters["b"], rpam.parameters["c"][2])
p_ellipse_l = gmsh.model.geo.addPoint(rpam.parameters["c"][0] - rpam.parameters["a"], rpam.parameters["c"][1], rpam.parameters["c"][2])
p_ellipse_b = gmsh.model.geo.addPoint(rpam.parameters["c"][0], rpam.parameters["c"][1] - rpam.parameters["b"], rpam.parameters["c"][2])
gmsh.model.geo.synchronize()

ellipse_arc_rt = gmsh.model.geo.addEllipseArc(p_ellipse_r, p_ellipse_c, p_ellipse_r, p_ellipse_t)
ellipse_arc_tl = gmsh.model.geo.addEllipseArc(p_ellipse_t, p_ellipse_c, p_ellipse_r, p_ellipse_l)
ellipse_arc_lb = gmsh.model.geo.addEllipseArc(p_ellipse_l, p_ellipse_c, p_ellipse_r, p_ellipse_b)
ellipse_arc_br = gmsh.model.geo.addEllipseArc(p_ellipse_b, p_ellipse_c, p_ellipse_r, p_ellipse_r)
gmsh.model.geo.synchronize()

loop_ellipse = gmsh.model.geo.addCurveLoop([ellipse_arc_rt, ellipse_arc_tl, ellipse_arc_lb, ellipse_arc_br])
gmsh.model.geo.synchronize()

# add circle
p_circle_r = gmsh.model.geo.addPoint(rpam.parameters["c"][0] + rpam.parameters["r"], rpam.parameters["c"][1], rpam.parameters["c"][2])
p_circle_t = gmsh.model.geo.addPoint(rpam.parameters["c"][0], rpam.parameters["c"][1] + rpam.parameters["r"], rpam.parameters["c"][2])
p_circle_l = gmsh.model.geo.addPoint(rpam.parameters["c"][0] - rpam.parameters["r"], rpam.parameters["c"][1], rpam.parameters["c"][2])
p_circle_b = gmsh.model.geo.addPoint(rpam.parameters["c"][0], rpam.parameters["c"][1] - rpam.parameters["r"], rpam.parameters["c"][2])
gmsh.model.geo.synchronize()

circle_arc_rt = gmsh.model.geo.addCircleArc(p_circle_r, p_ellipse_c,  p_circle_t)
circle_arc_tl = gmsh.model.geo.addCircleArc(p_circle_t, p_ellipse_c, p_circle_l)
circle_arc_lb = gmsh.model.geo.addCircleArc(p_circle_l, p_ellipse_c, p_circle_b)
circle_arc_br = gmsh.model.geo.addCircleArc(p_circle_b, p_ellipse_c,  p_circle_r)
gmsh.model.geo.synchronize()

loop_circle = gmsh.model.geo.addCurveLoop([circle_arc_rt, circle_arc_tl, circle_arc_lb, circle_arc_br])
gmsh.model.geo.synchronize()


surface_square_minus_ellipse = gmsh.model.geo.addPlaneSurface([loop_out, loop_ellipse])
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, [ellipse_arc_rt, ellipse_arc_tl, ellipse_arc_lb, ellipse_arc_br], 2, surface_square_minus_ellipse)
gmsh.model.geo.synchronize()

surface_ellipse = gmsh.model.geo.addPlaneSurface([loop_ellipse, loop_circle])
gmsh.model.geo.synchronize()






# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# square lines
gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_sub_mesh_1_b_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_sub_mesh_1_b_id"], "line_out_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_sub_mesh_1_r_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_sub_mesh_1_r_id"], "line_out_23")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_sub_mesh_1_t_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_sub_mesh_1_t_id"], "line_out_34")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_sub_mesh_1_l_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_sub_mesh_1_l_id"], "line_out_41")


#ellipse loop
gmsh.model.addPhysicalGroup(1, [lines[i][1] for i in range(4, 8)], rpam.parameters["ellipse_loop_id"])
gmsh.model.setPhysicalName(1, rpam.parameters["ellipse_loop_id"], "ellipse_loop")

#circle loop
gmsh.model.addPhysicalGroup(1, [lines[i][1] for i in range(8, 12)], rpam.parameters["circle_loop_id"])
gmsh.model.setPhysicalName(1, rpam.parameters["circle_loop_id"], "circle_loop")


# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["sub_mesh_1_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["sub_mesh_1_id"], "surface_out")

gmsh.model.addPhysicalGroup(surfaces[1][0], [surfaces[1][1]], rpam.parameters["sub_mesh_0_id"])
gmsh.model.setPhysicalName(surfaces[1][0], rpam.parameters["sub_mesh_0_id"], "surface_in")

# set the resolution
# se resolution equal to parameters["resolution"] at buth distance 0 from surface_in, and  at distance max(rpam.parameters["L"],rpam.parameters["h"]) from sub_mesh_1_id
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [loop_circle])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"]/2)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(rpam.parameters["r"], rpam.parameters["a"]))

gmsh.model.mesh.field.setNumbers(distance, "FacesList", [loop_ellipse])

threshold_out = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_out, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold_out, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold_out, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold_out, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold_out, "DistMax", max(rpam.parameters["L"], rpam.parameters["h"]))


minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], mesh_metadata, output_directory, True)

msh.generate_sub_mesh(output_directory, os.path.join(output_directory, 'sub_meshes', 'in'), rpam.parameters["sub_mesh_0_id"])
msh.generate_sub_mesh(output_directory, os.path.join(output_directory, 'sub_meshes', 'out'), rpam.parameters["sub_mesh_1_id"])


# print the boundary points of the boundaries given by the ellipse and circle
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    [rpam.parameters['ellipse_loop_id']],
    os.path.join(output_directory, 'boundary_points_id_' + str(rpam.parameters['ellipse_loop_id']) + '.csv'))

msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    [rpam.parameters['circle_loop_id']],
    os.path.join(output_directory, 'boundary_points_id_' + str(rpam.parameters['circle_loop_id']) + '.csv'))

msh.clear_gmsh()
