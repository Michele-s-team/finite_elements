from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# read the lines
cf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
# read the vertices
sf = msh.read_mesh_components(lmsh.mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])  # Line measure
dp_l = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters['point_l_id'])  # Point measure for points at the edges of the mesh
dp_r = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters['point_r_id'])  # Point measure for points at the edges of the mesh

dp_lr = dp_l + dp_r

import check_mesh_tags_line

print(f'Module {__file__} called {check_mesh_tags_line.__file__}', flush=True)

boundary = 'on_boundary'
boundary_l = f'near(x[0], {parameters["x_l"]})'
boundary_r = f'near(x[1], {parameters["x_r"]})'
