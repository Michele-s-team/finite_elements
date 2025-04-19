'''
This code reads the mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh.py solution
'''
import argparse
import colorama as col
from dolfin import *
from fenics import *
from mshr import *
import numpy as np
import sys


# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import geometry as geo
import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

# CHANGE PARAMETERS HERE
L = 0.5
h = 0.5
c_r = [L / 2, h / 2]
r = 0.05

c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE

p_O_id = 1
surface_id = 1

# read the mesh
mesh = msh.read_mesh(args.input_directory + "/triangle_mesh.xdmf")

# read the triangles
vf = msh.read_mesh_components(mesh, 2, args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(mesh, 1, args.input_directory + "/line_mesh.xdmf")
# read the vertices
# sf = msh.read_mesh_components(mesh, 0, args.input_directory + "/vertex_mesh.xdmf")


dx = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=surface_id)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=5)
ds_circle = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=6)

ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle


'''
dline_12 = Measure( "ds", domain=mesh, subdomain_data=cf, subdomain_id=line_12_id )
dline_34 = Measure( "dS", domain=mesh, subdomain_data=cf, subdomain_id=line_34_id )
darc_21 = Measure( "ds", domain=mesh, subdomain_data=cf, subdomain_id=arc_21_id )
dp_2 = Measure( "dP", domain=mesh, subdomain_data=sf, subdomain_id=p_2_id )
'''


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

integral_exact_dx = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [L, h]) - \
                    cal.surface_integral_disk(function_test_integrals, r, c_r)

integral_exact_ds_l = cal.curve_integral_line(function_test_integrals, [0, 0], [0, h])
integral_exact_ds_r = cal.curve_integral_line(function_test_integrals, [L, 0], [L, h])
integral_exact_ds_t = cal.curve_integral_line(function_test_integrals, [0, h], [L, h])
integral_exact_ds_b = cal.curve_integral_line(function_test_integrals, [0, 0], [L, 0])

integral_exact_ds_circle = cal.curve_integral_circle(function_test_integrals, r, c_r)

integral_exact_ds_lr = integral_exact_ds_l + integral_exact_ds_r
integral_exact_ds_tb = integral_exact_ds_t + integral_exact_ds_b

integral_exact_ds_square = integral_exact_ds_lr + integral_exact_ds_tb

integral_exact_ds = integral_exact_ds_square + integral_exact_ds_circle


msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, dx, '\int f dx')

msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, ds_l, '\int f ds_l')
msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, ds_r, '\int f ds_r')
msh.test_mesh_integral(integral_exact_ds_t, function_test_integrals_fenics, ds_t, '\int f ds_t')
msh.test_mesh_integral(integral_exact_ds_b, function_test_integrals_fenics, ds_b, '\int f ds_b')

msh.test_mesh_integral(integral_exact_ds_lr, function_test_integrals_fenics, ds_lr, '\int f ds_lr')
msh.test_mesh_integral(integral_exact_ds_tb, function_test_integrals_fenics, ds_tb, '\int f ds_tb')

msh.test_mesh_integral(integral_exact_ds_square, function_test_integrals_fenics, ds_square, '\int f ds_square')

msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, ds, '\int f ds')
