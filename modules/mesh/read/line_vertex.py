from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg


# read the lines
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.h5", "cf")
# read the vertices
vf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, io.add_trailing_slash(rarg.args.input_directory) + "vertex_mesh.h5", "vf")



# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters['vertex_l_id'])
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters['vertex_r_id'])
ds_m = Measure("dS", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters['vertex_m_id'])
ds_lr = Measure("ds", domain=lmsh.mesh)


import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

boundary = 'on_boundary'
boundary_l = f'near(x[0], {parameters["x_l"]})'
boundary_r = f'near(x[0], {parameters["x_r"]})'
boundary_m = f'near(x[0], {parameters["x_m"]})'
