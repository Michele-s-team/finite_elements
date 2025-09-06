'''
this code defined a function used to test the mesh tags:
the mesh tags are tested by integrating over the tagged elements of the mesh this function
'''

from fenics import *
import numpy as np

import mesh.load as lmsh
import differential_geometry.manifold.geometry as geo

# CHANGE PARAMETERS HERE
# extract the first d components of c_test, where d is the mesh dimension, in order to make a test for the d-dimensional mesh under consideration 
c_test = [0.3, 0.76, 1.23][:lmsh.mesh.topology().dim()]
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define function_test_integrals_fenics
Q = FunctionSpace(lmsh.mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.np.linalg.norm(np.subtract(x, c_test)) - r_test) ** 2.0)


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):

        # here x may be a three- or two-dimensional array whose last entries are set to zero -> I give it the right dimension by doing x[:(lmsh.mesh.topology().dim())]
        values[0] = function_test_integrals(x[:(lmsh.mesh.topology().dim())])
    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q.ufl_element()))
