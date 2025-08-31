import colorama as col
from fenics import *
import importlib
import scipy.integrate as spi

import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.line_vertex')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

'''
# CHANGE PARAMETERS HERE
c_test = 0.76
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define tf.function_test_integrals_fenics
Q_test = FunctionSpace(lmsh.mesh, 'P', 2)


# tf.function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def tf.function_test_integrals(x):
    return (np.cos(np.abs(x - c_test) - r_test) ** 2.0)


# tf.function_test_integrals_fenics is the same as tf.function_test_integrals, but in fenics format
tf.function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = tf.function_test_integrals(x[0])

    def value_shape(self):
        return (1,)


tf.function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))
'''

integral_exact_dx = spi.quad(tf.function_test_integrals, 0, rmsh.parameters['L'])[0]
integral_exact_dx_l = spi.quad(tf.function_test_integrals, 0, rmsh.parameters['x_p'])[0]
integral_exact_dx_r = spi.quad(tf.function_test_integrals, rmsh.parameters['x_p'], rmsh.parameters['L'])[0]

test_mesh_integral_errors =  dict([])

test_mesh_integral_errors['\int dx f'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int dx f')

test_mesh_integral_errors['\int_{line l} dx f'] = msh.test_mesh_integral(integral_exact_dx_l, tf.function_test_integrals_fenics, rmsh.dx(1), '\int_{line l} dx f')
test_mesh_integral_errors['\int_{line r} dx f'] = msh.test_mesh_integral(integral_exact_dx_r, tf.function_test_integrals_fenics, rmsh.dx(2), '\int_{line r} dx f')

test_mesh_integral_errors['\int_{point_l} dp f'] = msh.test_mesh_integral(tf.function_test_integrals(0), tf.function_test_integrals_fenics, rmsh.dp_boundary(3), '\int_{point_l} dp f')
test_mesh_integral_errors['\int_{point_r} dp f'] = msh.test_mesh_integral(tf.function_test_integrals(rmsh.parameters['L']), tf.function_test_integrals_fenics, rmsh.dp_boundary(4), '\int_{point_r} dp f')
test_mesh_integral_errors['\int_{point_in} dp f'] = msh.test_mesh_integral(tf.function_test_integrals(rmsh.parameters['x_p']), tf.function_test_integrals_fenics, rmsh.dp_bulk(5), '\int_{point_in} dp f')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
