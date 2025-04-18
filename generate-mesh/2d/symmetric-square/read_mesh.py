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
import scipy.integrate as integrate
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
L = 1
h = 1
c_r = [L / 2, h / 2]
r = 0.3
c_test = [0.3, 0.76]
r_test = 0.345


# CHANGE PARAMETERS HERE

def function_test_integral_expression(x):
    return np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0
    # return 1

# remember that this function takes y as first argument, x as second argument
def function_test_integral(y, x):
    return function_test_integral_expression([x, y])

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = 1
        values[0] = function_test_integral_expression(x)
        # values[0] = x[0]

    def value_shape(self):
        return (1,)





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
ds_r = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=2)
ds_t = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=3)
ds_b = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=1)
ds_l = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=4)
ds_circle = Measure("ds", domain=mesh, subdomain_data=cf, subdomain_id=5)

Q = FunctionSpace(mesh, 'P', 1)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test = Function(Q)
f_test.interpolate(FunctionTestIntegralExpression(element=Q.ufl_element()))

# compute exact integrals
integral_exact_dx = (integrate.dblquad(function_test_integral, -0, L, lambda x: 0, lambda x: h)[0] -
                     integrate.dblquad(lambda rho, theta: rho * function_test_integral(c_r[1] + rho * np.sin(theta),
                                                                                       c_r[0] + rho * np.cos(theta)), 0,
                                       2 * np.pi, lambda rho: 0, lambda rho: r)[0])
integral_exact_ds_l = (integrate.quad(lambda y: function_test_integral(y, 0), 0, h))[0]
integral_exact_ds_r = (integrate.quad(lambda y: function_test_integral(y, L), 0, h))[0]
integral_exact_ds_t = (integrate.quad(lambda x: function_test_integral(h, x), 0, L))[0]
integral_exact_ds_b = (integrate.quad(lambda x: function_test_integral(0, x), 0, L))[0]
integral_exact_ds_circle = \
(integrate.quad(lambda theta: r * function_test_integral(c_r[1] + r * np.sin(theta), c_r[0] + r * np.cos(theta)), 0, 2 * np.pi))[
    0]

msh.test_mesh_integral(integral_exact_dx, f_test, dx, '\int dx f')
# msh.test_mesh_integral(0.7765772342243651, f_test, ds_b, '\int ds_b f')
msh.test_mesh_integral(integral_exact_ds_t, f_test, ds_t, '\int ds_t f')
msh.test_mesh_integral(integral_exact_ds_b, f_test, ds_b, '\int ds_b f')
msh.test_mesh_integral(integral_exact_ds_l, f_test, ds_l, '\int ds_l f')
msh.test_mesh_integral(integral_exact_ds_r, f_test, ds_r, '\int ds_r f')
msh.test_mesh_integral(integral_exact_ds_circle, f_test, ds_circle, '\int ds_circle f')
