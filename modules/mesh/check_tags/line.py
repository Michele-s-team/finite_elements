import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh

rmsh = importlib.import_module('mesh.read.line')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# Global variables which will be set according to the gauge choice
solution_path = None

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

integral_exact_dx = cal.curve_integral_line(function_test_integrals, rmsh.parameters['x_l'], rmsh.parameters['x_r'])

integral_exact_ds_l = function_test_integrals_fenics(rmsh.parameters['x_l'])
integral_exact_ds_r = function_test_integrals_fenics(rmsh.parameters['x_r'])
integral_exact_ds = integral_exact_ds_l + integral_exact_ds_r

test_mesh_integral_errors =  dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_l'] = msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l')
test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
if solution_path is not None:
    io.write_parameters_to_csv_file(io.add_trailing_slash(solution_path) + 'integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
