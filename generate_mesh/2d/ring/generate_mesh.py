'''
generate a mesh given by a ring

Run it with
    python3 generate_mesh.py.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/ring"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle(rpam.parameters["c_r"], rpam.parameters["r"], mesh_size=rpam.parameters["resolution"])
circle_R = model.add_circle(rpam.parameters["c_R"], rpam.parameters["R"], mesh_size=rpam.parameters["resolution"])

plane_surface = model.add_plane_surface(circle_R.curve_loop, holes=[circle_r.curve_loop])

model.synchronize()
model.add_physical([plane_surface], "Volume")

# I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
model.add_physical(circle_r.curve_loop.curves, "Circle r")
model.add_physical(circle_R.curve_loop.curves, "Circle R")

geometry.generate_mesh(64)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)


msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

gmsh.clear()
geometry.__exit__()
