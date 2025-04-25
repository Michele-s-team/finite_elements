'''
This code reads the 1d mesh generated from generate_1dmesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_line_vertex_mesh.py [path where to find the mesh]
example:
clear; clear; python3 read_line_vertex_mesh.py /home/fenics/shared/generate_mesh/1d/line_vertex/solution

'''
from fenics import *
import numpy as np
import argparse
from dolfin import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh
import runtime_arguments as rarg

# CHANGE PARAMETERS HERE
L = 1
x_p = np.pi/8.0
# CHANGE PARAMETERS HERE

# read the mesh
mesh = msh.read_mesh(rarg.args.input_directory + "/line_mesh.xdmf")

# read the lines
cf = msh.read_mesh_components(mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# read the lines
sf = msh.read_mesh_components(mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = (np.cos(3 + x[0])) ** 2

    def value_shape(self):
        return (1,)


dx = Measure("dx", domain=mesh, subdomain_data=cf)  # Line measure
dp_boundary = Measure("ds", domain=mesh, subdomain_data=sf)  # Point measure for points at the edges of the mesh
dp_bulk = Measure("dS", domain=mesh, subdomain_data=sf)  # Point measure for points in the mesh

import check_mesh_tags_line_vertex
