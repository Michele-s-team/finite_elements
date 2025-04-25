'''
This code reads the mesh generated from generate_mesh_ring.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh_ring.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh_ring.py solution
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
c_r = [0,0]
c_R = [0,0]
r = 1
R = 2

c_test = [0.3, 0.76]
r_test = 0.345

surface_id = 1
circle_r_id = 2
circle_R_id = 3
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
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=circle_r_id)
ds_R = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=circle_R_id)


ds = ds_r + ds_R

# a function space used solely to define function_test_integrals_fenics
Q = FunctionSpace(mesh, 'P', 2)

function_test_symmetry = Function(Q)


# analytical expression for a  scalar function used to test the ds
class FunctionTestSymmetryExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]+x[1]

    def value_shape(self):
        return (1,)


function_test_symmetry.interpolate(FunctionTestSymmetryExpression(element=Q.ufl_element()))

import check_mesh_tags_ring

print(
    f'int f_test_symmetry = {col.Fore.YELLOW}{assemble(function_test_symmetry * dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}')
