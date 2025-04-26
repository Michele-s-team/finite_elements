'''
This code reads the mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh_square_no_circle.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh_square_no_circle.py solution
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
L = 2
h = 1
# CHANGE PARAMETERS HERE

# read the mesh
mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the triangles
vf = msh.read_mesh_components(mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

dx = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=1)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=5)

ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb

import check_mesh_tags_square_no_circle

