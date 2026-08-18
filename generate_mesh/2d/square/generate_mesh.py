'''
generate a mesh given by a square with a circular hole in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

msh.clear_gmsh(geometry)
