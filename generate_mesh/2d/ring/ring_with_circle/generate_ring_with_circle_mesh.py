'''
generate a mesh given by a ring with an inner circle embedded in the mesh

Run with
    clear; clear; python3 generate_mesh.py [path where to read the parameter file] [path where to store the solution]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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
output_directory = rarg.args.output_directory
output_directory = io.add_trailing_slash(output_directory)

mesh_file_name = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


# Initialize empty geometry using the built-in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add the inner and outer circles
circle_r = model.add_circle(rpam.parameters["c_r"], rpam.parameters["r"], mesh_size=rpam.parameters["resolution"])
circle_R = model.add_circle(rpam.parameters["c_R"], rpam.parameters["R"], mesh_size=rpam.parameters["resolution"])
circle_rho = model.add_circle(rpam.parameters["c_r"], rpam.parameters["rho"], mesh_size=rpam.parameters["resolution"])

plane_surface_r_rho = model.add_plane_surface(circle_rho.curve_loop, holes=[circle_r.curve_loop])
plane_surface_rho_R = model.add_plane_surface(circle_R.curve_loop, holes=[circle_rho.curve_loop])

model.synchronize()

# tag surfaces
model.add_physical([plane_surface_r_rho], "Ring Between r and rho")
model.add_physical([plane_surface_rho_R], "Ring Between rho and R")

# tag lines
model.add_physical(circle_r.curve_loop.curves, "Circle r")  # Inner circle (radius r)
model.add_physical(circle_rho.curve_loop.curves, "Circle rho")  # Intermediate circle (radius rho)
model.add_physical(circle_R.curve_loop.curves, "Circle R")  # Outer circle (radius R)

geometry.generate_mesh(64)
gmsh.write(mesh_file_name)

# Write the mesh and components to file
msh.print_mesh_edges_to_csv(mesh_file_name, output_directory + 'line_vertices.csv')

# Save the line and triangle mesh as usual
mesh_from_file = meshio.read(mesh_file_name)

msh.full_write(mesh_file_name, ['triangle', 'line'], metadata, output_directory, True)

# Clean up
gmsh.clear()
geometry.__exit__()
