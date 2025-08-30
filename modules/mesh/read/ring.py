from fenics import *

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

# test for surface elements
dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_R = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds = ds_r + ds_R

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.ring')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_r = f'on_boundary && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) < ({parameters["r"]} + {parameters["R"]})/2.0'
boundary_R = f'on_boundary && sqrt(pow(x[0] - {parameters["c_R"][0]}, 2) + pow(x[1] - {parameters["c_R"][1]}, 2)) > ({parameters["r"]} + {parameters["R"]})/2.0'
