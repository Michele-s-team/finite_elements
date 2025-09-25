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

# add the points which describe the l, r and b edge, and the parts of the t edge which surround the semi-circle, and the semi-circle
my_points = [gmsh.model.geo.addPoint(0, 0, 0),
             gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0),
             gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"] + rpam.parameters["r"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"], rpam.parameters["h"] - rpam.parameters["r"], 0),
             gmsh.model.geo.addPoint(rpam.parameters["c_r_x"] - rpam.parameters["r"], rpam.parameters["h"], 0),
             gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)
             ]


# Add lines between all points creating the rectangle
line_b = gmsh.model.geo.addLine(my_points[0], my_points[1])
line_r = gmsh.model.geo.addLine(my_points[1], my_points[2])
line_tr = gmsh.model.geo.addLine(my_points[2], my_points[3])
arc_r = gmsh.model.geo.addCircleArc(my_points[3], half_circle_center, my_points[4])
arc_l = gmsh.model.geo.addCircleArc(my_points[4], half_circle_center, my_points[5])
line_tl = gmsh.model.geo.addLine(my_points[5], my_points[6])
line_l = gmsh.model.geo.addLine(my_points[6], my_points[0])

loop = gmsh.model.geo.addCurveLoop([line_b, line_r, line_tr, arc_r, arc_l, line_tl, line_l])
plane_surface = gmsh.model.geo.addPlaneSurface([loop])

'''
model.synchronize()

gmsh.model.addPhysicalGroup(2, [plane_surface._id], tag=1)
gmsh.model.setPhysicalName(2, 1, "Volume")

# model.add_physical([plane_surface], "Volume", tag=rpam.parameters['surface_id'])
model.add_physical([line_b], "b")
model.add_physical([line_r], "r")
model.add_physical([line_l], "l")
model.add_physical([line_tl], "tl")
model.add_physical([line_tr], "tr")
model.add_physical([arc_l, arc_r], "half_circle")

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

gmsh.clear()
geometry.__exit__()
'''