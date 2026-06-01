'''
NOTE: this VP is well posed only of the degree of the function space of u is <= 2. If it is > 2, some DOFs sitting within the mesh triangles will appear, and these will not be constrained by the VP F
'''

from fenics import *
import importlib
import numpy as np
from scipy.optimize import brentq
import sys
import ufl as ufl

import calculus as cal
import constants.utils as const
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)


sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])



# 1. define expressions for BCs


'''
the curve is parametrized with 
r(t) = c + {a cos(2 pi t), b sin(2 pi t)}

and 0 < t < 1
'''
def X(t):
    return np.add(rmsh.parameters['c'], [rmsh.parameters['a'] * np.cos(2.0 * np.pi * t)/(2.0 + np.cos(2.0 * np.pi * t)),  rmsh.parameters['b'] * np.sin(2.0 * np.pi * t)])

'''
compute the difference between the polar angle of the curve vector \vec{X} writh respect to 'c' and a given angle
Input values: 
    - `t`: the curve parameter
    - `theta_0`: the given angle

Return values:
    - [theta of X(t)] - theta_0

'''
def delta_theta(t, theta_0):

    return cal.atan_quad([ X(t)[0] - rmsh.parameters['c'][0], X(t)[1] - rmsh.parameters['c'][1] ]) - theta_0

'''
return the parameter `t` of the curve `X` corresponding to a point `x` on the plane
Input values: 
    - `x`: the point
Return values: 
    - `t`: the value of the parametric coordinate of the curve such that `X(t)` forms the same polar angle as `x` with respect to `c`
'''
def t_X(x):

    # the polar angle that `x` forms with respect to `c``
    theta_x = cal.atan_quad([ x[0] - rmsh.parameters['c'][0], x[1] - rmsh.parameters['c'][1] ])

    try:
        # try to bracket the solution in the interval 0 < t < 1

        t = brentq(delta_theta, a=0, b=1.0, args=(theta_x) )

    except ValueError:
        # if the previous bracket fails, try to bracket the solution in the interval 0.5 < t < 1.5

        t = brentq(delta_theta, a=0.5, b=1.5, args=(theta_x))  
        
    return t


class e_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of `t` corresponding to `x`
        t = t_X(x)

        # t = 1.0/(2.0*np.pi) * cal.atan_quad([ (x[0] - rmsh.parameters['c'][0])/rmsh.parameters['a'], (x[1] - rmsh.parameters['c'][1])/rmsh.parameters['b'] ])

        values[0] = -4.0*np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * t) / (2.0 + np.cos(2.0 * np.pi * t))**2
        values[1] = 2.0*np.pi * rmsh.parameters['b'] * np.cos(2.0 * np.pi * t) 

    def value_shape(self):
        return (2,)
    
    
class n_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of `t` corresponding to `x`
        t = t_X(x)

        # t = 1.0/(2.0*np.pi) * cal.atan_quad([ (x[0] - rmsh.parameters['c'][0])/rmsh.parameters['a'], (x[1] - rmsh.parameters['c'][1])/rmsh.parameters['b'] ])

        norm = np.sqrt((4.0*np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * t) / (2.0 + np.cos(2.0 * np.pi * t))**2)**2 + (2.0*np.pi * rmsh.parameters['b'] * np.cos(2*np.pi*t))**2)

        values[0] = 2.0*np.pi * rmsh.parameters['b'] * np.cos(2*np.pi*t) / norm
        values[1] = 4.0*np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * t) / (2.0 + np.cos(2.0 * np.pi * t))**2 / norm

    def value_shape(self):
        return (2,)

class u_expression(UserExpression):
    def eval(self, values, x):

     
        values[0] = x[0]
        values[1] = -x[1]

    def value_shape(self):
        return (2,)

    
msh.interpolate_dg(fsp.e, e_expression())
msh.interpolate_dg(fsp.n, n_expression())
msh.interpolate_dg(fsp.u, u_expression())


bcs = []

# # variational problem

F_mu = (\
        (fsp.mu - 1.0/2.0 * (fsp.e[i].dx(j) * fsp.e[j] * fsp.n[i]) / dot(fsp.e, fsp.e)) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.mu, bgeo.facet_normal[0])[i] *  msh.jump(fsp.nu_mu, bgeo.facet_normal[0])[i] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )

F_grad_u = (\
        (fsp.grad_u[i, j] - fsp.u[i].dx(j)) * fsp.nu_grad_u[i, j] \
    ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.grad_u[i, j], bgeo.facet_normal[0])[k] *  msh.jump(fsp.nu_grad_u[i, j], bgeo.facet_normal[0])[k] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )

F = F_mu + F_grad_u