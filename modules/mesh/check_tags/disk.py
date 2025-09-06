import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.disk')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

'''
# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define tf.function_test_integrals_fenics
Q_test = FunctionSpace(lmsh.mesh, 'P', 2)


# tf.function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def tf.function_test_integrals(x):
    return (np.cos(geo.np.linalg.norm(np.subtract(x, c_test)) - r_test) ** 2.0)


# tf.function_test_integrals_fenics is the same as tf.function_test_integrals, but in fenics format
tf.function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = tf.function_test_integrals(x)

    def value_shape(self):
        return (1,)


tf.function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))
'''

integral_exact_dx = cal.surface_integral_disk(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"][:2])

integral_exact_ds = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"][:2])

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
