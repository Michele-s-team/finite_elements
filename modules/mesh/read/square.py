from fenics import *

import calculus as calc
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
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5)
ds_circle = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=6)
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

msh.check_mesh_symmetry(lmsh.mesh, parameters["c_r"])

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {parameters["L"]})'
boundary_lr = f'near(x[0], 0) || near(x[0], {parameters["L"]})'
boundary_tb = f'near(x[1], 0) || near(x[1], {parameters["h"]})'
boundary_square = f'on_boundary && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) > {(parameters["r"] + calc.min_dist_c_r_rectangle(parameters["L"], parameters["h"], parameters["c_r"])) / 2}'
boundary_circle = f'on_boundary && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) < {(parameters["r"] + calc.min_dist_c_r_rectangle(parameters["L"], parameters["h"], parameters["c_r"])) / 2}'
