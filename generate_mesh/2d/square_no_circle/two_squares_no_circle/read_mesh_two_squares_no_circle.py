'''
This code reads the mesh generated from generate_two_squares_mesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh_two_squares_no_circle.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh_two_squares_no_circle.py solution
'''

from dolfin import *
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh
import runtime_arguments as rarg

# CHANGE PARAMETERS HERE
L = 1
h = 2
L_m = L / 3

l_surface_id = 1
r_surface_id = 2
l_line_id = 3
lb_line_id = 4
rb_line_id = 5
r_line_id = 6
tr_line_id = 7
tl_line_id = 8
m_line_id = 9
# CHANGE PARAMETERS HERE

# read the mesh
mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the triangles
vf = msh.read_mesh_components(mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

dx_l = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=l_surface_id)
dx_r = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=r_surface_id)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=l_line_id)
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=r_line_id)
ds_lb = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=lb_line_id)
ds_rb = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=rb_line_id)
ds_rt = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=tr_line_id)
ds_lt = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=tl_line_id)
ds_m = Measure("dS", domain=mesh, subdomain_data=cf, subdomain_id=m_line_id)

dx = dx_l + dx_r

ds_b = ds_lb + ds_rb
ds_t = ds_lt + ds_rt

ds = ds_l + ds_r + ds_t + ds_b

# ds_square = ds_lr + ds_tb

import check_mesh_tags_two_squares_no_circle
