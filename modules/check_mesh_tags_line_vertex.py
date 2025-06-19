import colorama as col
import dolfin
from fenics import *
import numpy as np
import scipy.integrate as spi

import input_output as io
import load_mesh as lmsh
import mesh as msh
import read_mesh_line_vertex as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = 0.76
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(lmsh.mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(np.abs(x - c_test) - r_test) ** 2.0)


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x[0])

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

integral_exact_dx = spi.quad(function_test_integrals, 0, rmsh.parameters['L'])[0]
integral_exact_dx_l = spi.quad(function_test_integrals, 0, rmsh.parameters['x_p'])[0]
integral_exact_dx_r = spi.quad(function_test_integrals, rmsh.parameters['x_p'], rmsh.parameters['L'])[0]

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int dx f'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_l, function_test_integrals_fenics, rmsh.dx(1), '\int_{line l} dx f'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_r, function_test_integrals_fenics, rmsh.dx(2), '\int_{line r} dx f'))

test_mesh_integral_errors.append(msh.test_mesh_integral(function_test_integrals(0), function_test_integrals_fenics, rmsh.dp_boundary(3), '\int_{point_l} dp f'))
test_mesh_integral_errors.append(msh.test_mesh_integral(function_test_integrals(rmsh.parameters['L']), function_test_integrals_fenics, rmsh.dp_boundary(4), '\int_{point_r} dp f'))
test_mesh_integral_errors.append(msh.test_mesh_integral(function_test_integrals(rmsh.parameters['x_p']), function_test_integrals_fenics, rmsh.dp_bulk(5), '\int_{point_in} dp f'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
