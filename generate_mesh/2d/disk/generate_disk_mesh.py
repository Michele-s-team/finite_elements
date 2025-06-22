'''
generate a mesh given by a disk

Run it with
    python3 generate_disk_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/disk"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_disk_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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
output_directory = rarg.args.output_directory
output_directory = io.add_trailing_slash(output_directory)

mesh_file = output_directory + "mesh.msh"
mesh_metadata_file_name = rarg.args.output_directory + '/mesh_metadata.csv'

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle(rpam.parameters["c_r"], rpam.parameters["r"], mesh_size=rpam.parameters["resolution"])

plane_surface = model.add_plane_surface(circle_r.curve_loop, holes=[])

model.synchronize()
model.add_physical([plane_surface], "Volume")

# I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
model.add_physical(circle_r.curve_loop.curves, "Circle r")

geometry.generate_mesh(64)
gmsh.write(mesh_file)

msh.print_mesh_lines_to_csv(mesh_file, output_directory + 'line_vertices.csv')

gmsh.clear()
geometry.__exit__()

mesh_from_file = meshio.read(mesh_file)

# print line mesh
line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_directory + "line_mesh.xdmf", line_mesh)

# print triangle mesh
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)

# print  mesh vertices
mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

# print mesh metadata
io.write_parameters_to_csv_file(mesh_metadata_file_name, rpam.parameters)
