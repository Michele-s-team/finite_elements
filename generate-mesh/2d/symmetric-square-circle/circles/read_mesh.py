'''
This code reads the mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_mesh.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh.py solution
'''
import argparse
from dolfin import *
from fenics import *
from mshr import *
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import geometry as geo
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

# CHANGE PARAMETERS HERE


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


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = 1
        values[0] = function_test_integral_expression(x)
        # values[0] = x[0]

    def value_shape(self):
        return (1,)


def function_test_integral_expression(x):
    return np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0
    # return 1


dx = Measure("dx", domain=mesh, subdomain_data=vf, subdomain_id=surface_id)
ds_b = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=1)
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=2)
ds_t = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=3)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=4)
ds_circle = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=5)

'''
dline_12 = Measure( "ds", domain=mesh, subdomain_data=cf, subdomain_id=line_12_id )
dline_34 = Measure( "dS", domain=mesh, subdomain_data=cf, subdomain_id=line_34_id )
darc_21 = Measure( "ds", domain=mesh, subdomain_data=cf, subdomain_id=arc_21_id )
dp_2 = Measure( "dP", domain=mesh, subdomain_data=sf, subdomain_id=p_2_id )
'''

# dp_O = Measure("dP", domain=mesh, subdomain_data=sf, subdomain_id=p_O_id)

Q = FunctionSpace(mesh, 'P', 1)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test = Function(Q)
f_test.interpolate(FunctionTestIntegralExpression(element=Q.ufl_element()))

msh.test_mesh_integral(0.9063111599974194, f_test, dx, '\int dx f_surface')
# msh.test_mesh_integral(0.7932215226341747, f_test, dp_O, '\int dp f_{p_O}')
msh.test_mesh_integral(0.7765772342243651, f_test, ds_b, '\int ds_b f')
msh.test_mesh_integral(0.8056313961280863, f_test, ds_r, '\int ds_r f')
msh.test_mesh_integral(0.9756236687066221, f_test, ds_t, '\int ds_t f')
msh.test_mesh_integral(0.9620471547152678, f_test, ds_l, '\int ds_l f')
msh.test_mesh_integral(0.6251868570121668, f_test, ds_circle, '\int ds_circle f')

# msh.test_mesh_integral(0.687851524196998, f_test, ds_aux, '\int ds_{aux} f')

'''
msh.test_mesh_integral( 0.1588462551091818, f_test, dp_2, '\int dp f_{p_2}' )
msh.test_mesh_integral( cal.integral_2d_curve( function_test_integral_expression, curve_arc_21 ), f_test, darc_21, '\int dl f_{arc_21}' )
msh.test_mesh_integral( 0.652012217844941, f_test, dline_34, '\int dl f_{line_34}' )
'''
