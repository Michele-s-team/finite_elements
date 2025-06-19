'''
Ths code generates a 1d mesh given by a segment with a vertex in the segment

Run with
    clear; clear; python3 generate_mesh_line_vertex.py [path where to read the parameter file] [path where to store the solution]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/1d/line_vertex"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/1d/line_vertex/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_line_vertex.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh
import runtime_arguments as rarg
print(f'input_directory: {rarg.args.input_directory}\noutput_directory: {rarg.args.output_directory}')
import read_mesh_parameters as rmpam


mesh_file_name = rarg.args.output_directory + "/mesh.msh"
mesh_metadata_file_name = rarg.args.output_directory + '/mesh_metadata.csv'



# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 1d object a set of lines
points = [model.add_point((0, 0, 0), mesh_size=rmpam.parameters['resolution']),
          model.add_point((rmpam.parameters["x_p"], 0, 0), mesh_size=rmpam.parameters['resolution']),
          model.add_point((rmpam.parameters["L"], 0, 0), mesh_size=rmpam.parameters['resolution'])
          ]
my_lines = [model.add_line(points[0], points[1]), model.add_line(points[1], points[2])]

model.synchronize()

# print("# of lines added = ", len(my_lines))

model.add_physical([my_lines[0]], "line1")
model.add_physical([my_lines[1]], "line2")
model.add_physical([points[0]], "point_l")
model.add_physical([points[2]], "point_r")
model.add_physical([points[1]], "point_in")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file_name)

model.__exit__()

mesh_from_file = meshio.read(mesh_file_name)

# #create a tetrahedron mesh
# tetrahedron_mesh = create_mesh(mesh_from_file, "tetra", True)
# meshio.write("solution/tetrahedron_mesh.xdmf", tetrahedron_mesh)
# 
# #create a triangle mesh
# triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
# meshio.write("solution/triangle_mesh.xdmf", triangle_mesh)


# create a line mesh
line_mesh = msh.create_mesh(mesh_from_file, "line", True)
meshio.write(rarg.args.output_directory + "/line_mesh.xdmf", line_mesh)

# create a vertex mesh
vertex_mesh = msh.create_mesh(mesh_from_file, "vertex", True)
meshio.write(rarg.args.output_directory + "/vertex_mesh.xdmf", vertex_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(rarg.args.output_directory + "/line_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, rarg.args.output_directory + "/vertices.csv")

# print metadata
io.write_parameters_to_csv_file(mesh_metadata_file_name, [('L', rmpam.parameters['L']), ('x_p', rmpam.parameters['x_p']), ('resolution', rmpam.parameters['resolution'])])
