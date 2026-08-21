'''
generate a mesh given by a square with a 'dent' given by a half of a circle on its top edge. This is supposed to represent one half of a square with a circular hole. 

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/half_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/half_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


print(f'output_directory = "{output_directory}"')

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

half_circle_center = gmsh.model.geo.addPoint(rpam.parameters["c_r_x"], rpam.parameters["h"], 0)
gmsh.model.geo.synchronize()


# add the points which describe the l, r and b edge, and the parts of the t edge which surround the semi-circle, and the semi-circle
my_points = [gmsh.model.geo.addPoint(0, 0, 0),
             gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0),
             gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"] + rpam.parameters["r"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"], rpam.parameters["h"] - rpam.parameters["r"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"] - rpam.parameters["r"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)
             ]
gmsh.model.geo.synchronize()



# Add lines between all points creating the rectangle
line_b = gmsh.model.geo.addLine(my_points[0], my_points[1])
line_r = gmsh.model.geo.addLine(my_points[1], my_points[2])
line_tr = gmsh.model.geo.addLine(my_points[2], my_points[3])
arc_r = gmsh.model.geo.addCircleArc(my_points[3], half_circle_center, my_points[4])
arc_l = gmsh.model.geo.addCircleArc(my_points[4], half_circle_center, my_points[5])
line_tl = gmsh.model.geo.addLine(my_points[5], my_points[6])
line_l = gmsh.model.geo.addLine(my_points[6], my_points[0])
gmsh.model.geo.synchronize()

loop = gmsh.model.geo.addCurveLoop([line_b, line_r, line_tr, arc_r, arc_l, line_tl, line_l])
gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()


# tag the surface
gmsh.model.addPhysicalGroup(2, [surface], rpam.parameters['surface_id'])
gmsh.model.setPhysicalName(2, rpam.parameters['surface_id'], "surface")


# tag lines
gmsh.model.addPhysicalGroup(1, [line_l], rpam.parameters['line_l_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['line_l_id'], "line_l")

gmsh.model.addPhysicalGroup(1, [line_r], rpam.parameters['line_r_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['line_r_id'], "line_r")

gmsh.model.addPhysicalGroup(1, [line_tl], rpam.parameters['line_tl_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['line_tl_id'], "line_tl")

gmsh.model.addPhysicalGroup(1, [arc_l, arc_r], rpam.parameters['half_circle_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['half_circle_id'], "arc")

gmsh.model.addPhysicalGroup(1, [line_tr], rpam.parameters['line_tr_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['line_tr_id'], "line_tr")

gmsh.model.addPhysicalGroup(1, [line_b], rpam.parameters['line_b_id'])
gmsh.model.setPhysicalName(1, rpam.parameters['line_b_id'], "line_b")


# set the resolution
# se resolution equal to parameters["resolution"] at  distance 0 from surface, and to parameters["resolution"] at distance max(rpam.parameters["L"],rpam.parameters["h"]) from surface
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface])

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



# Mesh and write
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

msh.clear_gmsh()