import dolfin
from fenics import *
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_1d_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg


# read the lines
cf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
# read the vertices
sf = msh.read_mesh_components(lmsh.mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")

#radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

# CHANGE PARAMETERS HERE
L = 1
x_p = np.pi/8.0
# CHANGE PARAMETERS HERE


dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf)  # Line measure
dp_boundary = Measure("ds", domain=lmsh.mesh, subdomain_data=sf)  # Point measure for points at the edges of the mesh
dp_bulk = Measure("dS", domain=lmsh.mesh, subdomain_data=sf)  # Point measure for points in the mesh

import check_mesh_tags_line_vertex

print(f'Module {__file__} called {check_mesh_tags_line_vertex.__file__}', flush=True)

