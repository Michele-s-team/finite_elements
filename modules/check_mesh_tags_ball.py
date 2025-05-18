import colorama as col
import dolfin
from fenics import *
import numpy as np

import load_3d_mesh as lmsh
import mesh as msh

# the module read_mesh_square which is being called will be in the local folder, e.g., in steady-state-no-flow
import calculus as cal
import geometry as geo
import input_output as io
import read_mesh_ball as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76, 1.23]
r_test = 0.345
# CHANGE PARAMETERS HERE

Q = FunctionSpace(lmsh.mesh, 'P', 1)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)
    # return 1


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q.ufl_element()))

test_mesh_integral_errors = []

integral_exact_dv = cal.volume_integral_ball(function_test_integrals, rmsh.r, rmsh.c_r)

# print(f'integral_exact_dv = {integral_exact_dv}')

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dv, function_test_integrals_fenics, rmsh.dx, '\int_ball f dx'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(7.06579, f_test_ds, rmsh.ds, '\int_sphere f ds'))

# print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
