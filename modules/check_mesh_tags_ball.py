import colorama as col
import dolfin
from fenics import *
import numpy as np

import load_3d_mesh as lmsh
import mesh as msh

# the module read_mesh_square which is being called will be in the local folder, e.g., in steady-state-no-flow
import input_output as io
import read_mesh_ball as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        # values[0] = 1.0
        values[0] = (np.cos(3 * x[2] - 2 * x[1] + x[0])) ** 2

    def value_shape(self):
        return (1,)



Q = FunctionSpace(lmsh.mesh, 'P', 1)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function(Q)
f_test_ds.interpolate(FunctionTestIntegral(element=Q.ufl_element()))

test_mesh_integral_errors = []

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
test_mesh_integral_errors.append(msh.test_mesh_integral(2.06773, f_test_ds, rmsh.dv_custom, '\int_ball f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(7.06579, f_test_ds, rmsh.ds_custom, '\int_sphere f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

