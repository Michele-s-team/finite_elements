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

l_surface_id = 1
r_surface_id = 2
l_line_id = 3
lb_line_id = 4
rb_line_id = 5
r_line_id = 6
tr_line_id = 7
tl_line_id = 8
m_line_id = 9

dx_l = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=l_surface_id)
dx_r = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=r_surface_id)
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=l_line_id)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=r_line_id)
ds_lb = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lb_line_id)
ds_rb = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=rb_line_id)
ds_rt = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=tr_line_id)
ds_lt = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=tl_line_id)
ds_m = Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=m_line_id)

dx = dx_l + dx_r

ds_b = ds_lb + ds_rb
ds_t = ds_lt + ds_rt
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b

ds = ds_lr + ds_tb

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.two_squares_no_circle')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {parameters["L"]})'
boundary_lr = f'near(x[0], 0) || near(x[0], {parameters["L"]})'
boundary_tb = f'near(x[1], 0) || near(x[1], {parameters["h"]})'
