import colorama as col
from fenics import *
import numpy as np

import load_mesh as lmsh
import mesh as msh

import calculus as cal
import geometry as geo
import input_output as io
import read_mesh_box as rmsh

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

integral_exact_dx = cal.volume_integral_box(function_test_integrals, rmsh.parameters["L"])

integral_exact_ds_le = cal.surface_integral_rectangle(lambda r: function_test_integrals([0, r[0], r[1]]), [0, 0], [rmsh.parameters["L"][1], rmsh.parameters["L"][2]])
integral_exact_ds_ri = cal.surface_integral_rectangle(lambda r: function_test_integrals([rmsh.parameters["L"][0], r[0], r[1]]), [0, 0], [rmsh.parameters["L"][1], rmsh.parameters["L"][2]])
integral_exact_ds_to = cal.surface_integral_rectangle(lambda r: function_test_integrals([r[0], rmsh.parameters["L"][1], r[1]]), [0, 0], [rmsh.parameters["L"][0], rmsh.parameters["L"][2]])
integral_exact_ds_bo = cal.surface_integral_rectangle(lambda r: function_test_integrals([r[0], 0, r[1]]), [0, 0], [rmsh.parameters["L"][0], rmsh.parameters["L"][2]])
integral_exact_ds_fr = cal.surface_integral_rectangle(lambda r: function_test_integrals([r[0], r[1], rmsh.parameters["L"][2]]), [0, 0], [rmsh.parameters["L"][0], rmsh.parameters["L"][1]])
integral_exact_ds_ba = cal.surface_integral_rectangle(lambda r: function_test_integrals([r[0], r[1], 0]), [0, 0], [rmsh.parameters["L"][0], rmsh.parameters["L"][1]])

integral_exact_ds_leri = integral_exact_ds_le + integral_exact_ds_ri
integral_exact_ds_tobo = integral_exact_ds_to + integral_exact_ds_bo
integral_exact_ds_frba = integral_exact_ds_fr + integral_exact_ds_ba

integral_exact_ds = integral_exact_ds_leri + integral_exact_ds_tobo + integral_exact_ds_frba

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int_ball f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_le, function_test_integrals_fenics, rmsh.ds_le, '\int_le f ds'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_ri, function_test_integrals_fenics, rmsh.ds_ri, '\int_ri f ds'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_to, function_test_integrals_fenics, rmsh.ds_to, '\int_to f ds'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_bo, function_test_integrals_fenics, rmsh.ds_bo, '\int_bo f ds'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_fr, function_test_integrals_fenics, rmsh.ds_fr, '\int_fr f ds'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_ba, function_test_integrals_fenics, rmsh.ds_ba, '\int_ba f ds'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
