'''
This code reads the mesh generated from generate_ring_mesh.py and it creates dvs and dss from labelled components of the mesh

Run with
    clear; clear; python3 check_mesh_ring.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_ring.py solution
'''

import dolfin
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# CHANGE PARAMETERS HERE
r = 1
R = 2
c_r = [0, 0]
c_R = [0, 0]
# CHANGE PARAMETERS HERE

# read the mesh
# mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the triangles
vf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=1)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=2)
ds_R = Measure("ds", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=3)

ds = ds_r + ds_R

import check_mesh_tags_ring
