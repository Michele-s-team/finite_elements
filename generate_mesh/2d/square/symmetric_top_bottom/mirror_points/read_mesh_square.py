'''
This code reads the mesh generated from generate_mesh_square.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh_square.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh_square.py solution
'''
import colorama as col
from dolfin import *
from fenics import *
from mshr import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import runtime_arguments as rarg
import mesh as msh

# CHANGE PARAMETERS HERE
L = 1
h = 1
c_r = [L / 2, h / 2]
r = 0.25

c_test = [0.3, 0.76]
r_test = 0.345

surface_id = 1
l_edge_id = 2
r_edge_id = 3
t_edge_id = 4
b_edge_id = 5
circle_id = 6
# CHANGE PARAMETERS HERE


# read the mesh
mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the triangles
vf = msh.read_mesh_components(mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# read the vertices
# sf = msh.read_mesh_components(mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")

dx = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=surface_id)
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=r_edge_id)
ds_t = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=t_edge_id)
ds_b = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=b_edge_id)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=l_edge_id)
ds_circle = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=circle_id)

ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle

import check_mesh_tags_square
