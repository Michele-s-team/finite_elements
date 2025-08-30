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

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5)
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square_no_circle')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_l = 'near(x[0], 0.0)'
boundary_r = f'near(x[0], {parameters["L"]})'
boundary_t = f'near(x[1], {parameters["h"]})'
boundary_b = 'near(x[1], 0.0)'
boundary_lr = f'near(x[0], 0) || near(x[0], {parameters["L"]})'
boundary_tb = f'near(x[1], 0) || near(x[1], {parameters["h"]})'
