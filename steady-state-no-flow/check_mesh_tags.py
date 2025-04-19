from fenics import *
from mshr import *
import numpy as np
import scipy.integrate as integrate

import boundary_geometry as bgeo
import geometry as geo
import mesh as msh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
import read_mesh_square as rmsh


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

msh.test_mesh_integral(0.22908817224489927, f_test_ds, rmsh.dx, '\int f dx')
msh.test_mesh_integral(1.8449287777896068, f_test_ds, rmsh.ds_square, '\int_square f ds')
msh.test_mesh_integral(0.3049366444861381, f_test_ds, rmsh.ds_circle, '\int_circle f ds')
msh.test_mesh_integral(0.9336461710791771, f_test_ds, rmsh.ds_lr, '\int_lr f ds')
msh.test_mesh_integral(0.9112826067104298, f_test_ds, rmsh.ds_tb, '\int_tb f ds')

msh.test_mesh_integral(0.4625165259025798, f_test_ds, rmsh.ds_l, '\int_l f ds')
msh.test_mesh_integral(0.47112964517659733, f_test_ds, rmsh.ds_r, '\int_r f ds')
msh.test_mesh_integral(0.4982661696490371, f_test_ds, rmsh.ds_t, '\int_t f ds')
msh.test_mesh_integral(0.41301643706139274, f_test_ds, rmsh.ds_b, '\int_b f ds')
