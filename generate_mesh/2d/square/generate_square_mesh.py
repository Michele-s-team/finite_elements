'''
generate a mesh given by a square with a circular hole in it

Run it with
    python3 generate_square_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import meshio
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

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

print(f'output_directory = "{output_directory}"')

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

my_points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((0, rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"])]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line(my_points[i], my_points[i + 1])
                 for i in range(-1, len(my_points) - 1)]

channel_loop = model.add_curve_loop(channel_lines)

circle_r = model.add_circle(rpam.parameters["c_r"], rpam.parameters["r"], mesh_size=rpam.parameters["resolution"])

plane_surface = model.add_plane_surface(channel_loop, holes=[circle_r.curve_loop])

model.synchronize()

model.add_physical([plane_surface], "Volume")
model.add_physical([channel_lines[0]], "i")
model.add_physical([channel_lines[2]], "o")
model.add_physical([channel_lines[3]], "t")
model.add_physical([channel_lines[1]], "b")
model.add_physical(circle_r.curve_loop.curves, "c")

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

# write line mesh
msh.write_mesh_components(mesh_file, output_directory + "line_mesh.xdmf", "line", True)

# write triangle mesh
msh.write_mesh_components(mesh_file, output_directory + "triangle_mesh.xdmf", "triangle", True)

# print mesh vertices to csv file
mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

# print mesh lines to csv file
msh.print_mesh_lines_to_csv(mesh_file, output_directory + 'line_vertices.csv')

# print mesh metadata to csv file
io.write_parameters_to_csv_file(mesh_metadata_file_name, rpam.parameters)

gmsh.clear()
geometry.__exit__()
