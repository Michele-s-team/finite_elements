import colorama as col
from fenics import *

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import numpy as np
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")




focus = np.subtract(parameters["c"], [np.sqrt(parameters["a"] ** 2 - parameters["b"] ** 2), 0, 0])

print(f"Radius of mesh cell = {col.Fore.BLUE}{r_mesh}{col.Style.RESET_ALL}")



# test for surface elements
dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5)
ds_ellipse = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=6)
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds_l_tb_ellipse = ds_l + ds_t + ds_b + ds_ellipse
ds = ds_square + ds_ellipse

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square_ellipse')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

msh.check_mesh_symmetry(lmsh.mesh, parameters["c"])

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {parameters["L"]})'
boundary_lr = f'near(x[0], 0) || near(x[0], {parameters["L"]})'
boundary_tb = f'near(x[1], 0) || near(x[1], {parameters["h"]})'
boundary_square = f'on_boundary && (near(x[0], 0) || near(x[0], {parameters["L"]}) || near(x[1], 0) || near(x[1], {parameters["h"]}))'
boundary_ellipse = f'on_boundary && (!near(x[0], 0))  && (!near(x[0], {parameters["L"]})) && (!near(x[1], 0)) && (!near(x[1], {parameters["h"]}))'
