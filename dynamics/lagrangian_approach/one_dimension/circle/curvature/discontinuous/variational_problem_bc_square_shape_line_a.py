from fenics import *
import importlib
import numpy as np
from scipy.optimize import brentq
import ufl as ufl

import calculus as cal
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m, n, o, p, q, r, s, t, u = ufl.indices(13)


sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])



# 1. define expressions for BCs


'''
the curve y_s in the reference configuration and its derivative 
Input values: 
    - 's' :  the parametric coordinate 0 < s < 1
Return values; 
    - 'y_s', 'd y_s/ds': y(s) and d y(s) / ds
'''
def y_s_dy_ds(s):
    return np.add(rmsh.parameters['c'], [rmsh.parameters['a'] * np.cos(2.0 * np.pi * s)/(2.0 + np.cos(2.0 * np.pi * s)),  rmsh.parameters['b'] * np.sin(2.0 * np.pi * s)]), np.array([-4.0*np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * s) / (2.0 + np.cos(2.0 * np.pi * s))**2, 2.0*np.pi * rmsh.parameters['b'] * np.cos(2.0 * np.pi * s) ])

'''
difference between the polar angle of the curve vector y_s with respect to the point 'c' and a given angle theta_0
Input values: 
    - `s`: the curve parameter
    - `theta_0`: the given angle

Return values:
    - [polar angle of y(s)] - theta_0

'''
def delta_theta(s, theta_0):

    return cal.atan_quad([ y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ]) - theta_0


'''
return the parameter `s` of the curve `y(s)` corresponding to a point `y` on the plane
Input values: 
    - `y`: the point in the reference configuration
Return values: 
    - `s`: the value of the parametric coordinate of the curve `y(s)` such that the point y(s) forms the same polar angle as `y` with respect to `c`
'''
def s_y(y):

    # the polar angle that `y` forms with respect to `c``
    theta_y = cal.atan_quad([ y[0] - rmsh.parameters['c'][0], y[1] - rmsh.parameters['c'][1] ])

    try:
        # try to bracket the solution in the interval 0 < t < 1

        s = brentq(delta_theta, a=0, b=1.0, args=(theta_y) )

    except ValueError:
        # if the previous bracket fails, try to bracket the solution in the interval 0.5 < t < 1.5

        s = brentq(delta_theta, a=0.5, b=1.5, args=(theta_y))  
        
    return s



class f_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of the parametric coordinate `s` corresponding to `x`
        s = s_y(x)

        values[0] = y_s_dy_ds(s)[1][0]
        values[1] = y_s_dy_ds(s)[1][1]

    def value_shape(self):
        return (2,)
    


    
class nu_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of `t` corresponding to `x`
        s = s_y(x)

        _, d_y_s_ds = y_s_dy_ds(s)

        norm = np.linalg.norm(d_y_s_ds)

        values[0] = d_y_s_ds[1] / norm
        values[1] = - d_y_s_ds[0] / norm

    def value_shape(self):
        return (2,)
    


class u_expression(UserExpression):
    def eval(self, values, x):

     
        values[0] = rpam.parameters['d'] * x[0]
        values[1] = -rpam.parameters['d'] * x[1]

    def value_shape(self):
        return (2,)

    
msh.interpolate_dg(fsp.f, f_expression())
msh.interpolate_dg(fsp.nu, nu_expression())
msh.interpolate_dg(fsp.u, u_expression())


bcs = []

# # variational problem

F_mu = (\
        (fsp.mu \
        - (fsp.f[i] + fsp.grad_u[i, k] * fsp.f[k]).dx(j) * fsp.f[j] \
        * ( sqrt( dot(fsp.f, fsp.f) / (ela.F(fsp.u)[p, q] * ela.F(fsp.u)[p, r] * fsp.f[q] * fsp.f[r] ) ) \
        * bgeo.epsilon[i, s] * ela.F(fsp.u)[s, t] * bgeo.epsilon[t, u] * fsp.nu[u] )  \
        / (2.0 * (fsp.f[m] + fsp.grad_u[m, n] * fsp.f[n]) * (fsp.f[m] + fsp.grad_u[m, o] * fsp.f[o]) ) \
        ) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.mu, bgeo.facet_normal[0])[i] *  msh.jump(fsp.nu_mu, bgeo.facet_normal[0])[i] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )

F_grad_u = ( (fsp.grad_u[i, j] - fsp.u[i].dx(j)) * fsp.nu_grad_u[i, j] ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.grad_u[i, j], bgeo.facet_normal[0])[k] *  msh.jump(fsp.nu_grad_u[i, j], bgeo.facet_normal[0])[k] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )

F = F_mu + F_grad_u

# sign
