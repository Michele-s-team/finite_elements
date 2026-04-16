from fenics import *
import importlib

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# 1. read mesh quantities

# 1.1 read  triangles
sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")

# 1.2 read  lines

# 1.2.1 read boundary lines
mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# 1.2.2 read inner (I) lines
mf_I = msh.read_mesh_internal_components(lmsh.mesh, sf, lmsh.parameters['l_surface_id'], lmsh.parameters['r_surface_id'], lmsh.parameters['m_line_id'])


# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()


#2. define measure
dx_l = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['l_surface_id'])
dx_r = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['r_surface_id'])
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['l_line_id'])
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['r_line_id'])
ds_lb = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['lb_line_id'])
ds_rb = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['rb_line_id'])
ds_rt = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['tr_line_id'])
ds_lt = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=lmsh.parameters['tl_line_id'])
dS_l = Measure("dS", domain=lmsh.mesh, subdomain_data=mf_I, subdomain_id=lmsh.parameters['l_surface_id'])
dS_r = Measure("dS", domain=lmsh.mesh, subdomain_data=mf_I, subdomain_id=lmsh.parameters['r_surface_id'])
dS_m = Measure("dS", domain=lmsh.mesh, subdomain_data=mf_I, subdomain_id=lmsh.parameters['m_line_id'])

dx = dx_l + dx_r

ds_b = ds_lb + ds_rb
ds_t = ds_lt + ds_rt
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b

ds = ds_lr + ds_tb

# all internal facets contained into the square
dS = dS_l + dS_r + dS_m

check_mesh_module = importlib.import_module('mesh.check_tags.two_squares_no_circle')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

#3.  Define boundaries and obstacle
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {lmsh.parameters["L"]})'
boundary_lr = f'near(x[0], 0) || near(x[0], {lmsh.parameters["L"]})'
boundary_tb = f'near(x[1], 0) || near(x[1], {lmsh.parameters["h"]})'
