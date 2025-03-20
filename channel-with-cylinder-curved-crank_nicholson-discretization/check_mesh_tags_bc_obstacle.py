from fenics import *
from mshr import *
import numpy as np
import scipy.integrate as integrate

import boundary_geometry as bgeo
import geometry as geo
import mesh as msh
import read_mesh_bc_obstacle as rmsh

c_test = [0.3, 0.76]
r_test = 0.345


# remember that this function takes y as first argument, x as second argument
def test_function(y, x):
    return (np.cos( geo.my_norm( np.subtract( [x, y], c_test ) ) - r_test ) ** 2.0)


# a function space used solely to define f_test_ds
Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        values[0] = test_function( x[1], x[0] )

    def value_shape(self):
        return (1,)


f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )

integral_exact_dx = integrate.dblquad( test_function, 0, rmsh.L, lambda x: 0, lambda x: rmsh.h )[0] \
                    - integrate.dblquad( lambda r, theta: r * test_function( rmsh.c_r[1] + r * np.sin( theta ), rmsh.c_r[0] + r * np.cos( theta ) ), 0, 2 * np.pi, lambda r: 0, lambda r: rmsh.r )[0]

integral_exact_ds_l = (integrate.quad( lambda y: test_function( y, 0 ), 0, rmsh.h ))[0]
integral_exact_ds_r = (integrate.quad( lambda y: test_function( y, rmsh.L ), 0, rmsh.h ))[0]
integral_exact_ds_t = (integrate.quad( lambda x: test_function( rmsh.h, x ), 0, rmsh.L ))[0]
integral_exact_ds_b = (integrate.quad( lambda x: test_function( 0, x ), 0, rmsh.L ))[0]
integral_exact_ds_circle = (integrate.quad( lambda theta: rmsh.r * test_function( rmsh.c_r[1] + rmsh.r * np.sin( theta ), rmsh.c_r[0] + rmsh.r * np.cos( theta ) ), 0, 2 * np.pi ))[0]

integral_exact_ds_lr = integral_exact_ds_l + integral_exact_ds_r
integral_exact_ds_tb = integral_exact_ds_t + integral_exact_ds_b
integral_exact_ds_square = integral_exact_ds_lr + integral_exact_ds_tb

integral_exact_ds = integral_exact_ds_square + integral_exact_ds_circle

msh.test_mesh_integral( integral_exact_dx, f_test_ds, rmsh.dx, '\int f dx' )

msh.test_mesh_integral( integral_exact_ds_l, f_test_ds, rmsh.ds_l, '\int f ds_l' )
msh.test_mesh_integral( integral_exact_ds_r, f_test_ds, rmsh.ds_r, '\int f ds_r' )
msh.test_mesh_integral( integral_exact_ds_t, f_test_ds, rmsh.ds_t, '\int f ds_t' )
msh.test_mesh_integral( integral_exact_ds_b, f_test_ds, rmsh.ds_b, '\int f ds_b' )

msh.test_mesh_integral( integral_exact_ds_lr, f_test_ds, rmsh.ds_lr, '\int f ds_lr' )
msh.test_mesh_integral( integral_exact_ds_tb, f_test_ds, rmsh.ds_tb, '\int f ds_tb' )

msh.test_mesh_integral( integral_exact_ds_square, f_test_ds, rmsh.ds_square, '\int f ds_square' )
msh.test_mesh_integral( integral_exact_ds_circle, f_test_ds, rmsh.ds_circle, '\int f ds_circle' )

msh.test_mesh_integral( integral_exact_ds, f_test_ds, rmsh.ds, '\int f ds' )
