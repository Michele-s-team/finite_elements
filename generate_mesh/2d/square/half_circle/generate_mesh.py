'''
generate a mesh given by a square with a 'dent' given by a half of a circle on its top edge. This is supposed to represent one half of a square with a circular hole. 

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/half_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/half_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import meshio
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

half_circle_center = model.add_point((rpam.parameters["c_r_x"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"])

# add the points which describe the l, r and b edge, and the parts of the t edge which surround the semi-circle, and the semi-circle
my_points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["c_r_x"] + rpam.parameters["r"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["c_r_x"], rpam.parameters["h"] - rpam.parameters["r"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["c_r_x"] - rpam.parameters["r"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((0, rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"])
             ]


# Add lines between all points creating the rectangle
line_b = model.add_line(my_points[0], my_points[1])
line_r = model.add_line(my_points[1], my_points[2])
line_tr = model.add_line(my_points[2], my_points[3])
arc_r = model.add_circle_arc(my_points[3], half_circle_center, my_points[4])
arc_l = model.add_circle_arc(my_points[4], half_circle_center, my_points[5])
line_tl = model.add_line(my_points[5], my_points[6])
line_l = model.add_line(my_points[6], my_points[0])

channel_lines = [line_b, line_r, line_tr,
                 arc_r, arc_l,
                 line_tl, line_l
                 ]

channel_loop = model.add_curve_loop(channel_lines)


plane_surface = model.add_plane_surface(channel_loop)

model.synchronize()

model.add_physical([plane_surface], "Volume")
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
