from fenics import *
import importlib
import numpy as np
from scipy.optimize import brentq
import ufl as ufl


import calculus as cal
import constants.utils as const
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)

i, j, k, l, m, n, o, p, q, r, s, t, u = ufl.indices(13)


sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])









# 



# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = cal.atan_quad([ sh.y_s_dy_ds(0)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(0)[0][1] - rmsh.parameters['c'][1] ])
print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# solve for s_0
# REVISE: INCLUDE A SYSTEMATIC WAY TO DETERMINE THE INTERVAL WHERE TO LOOK FOR S_0

s_0 = brentq(lambda s:  np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=0.75, b=1.0 )

print(f's_0 = {s_0}\t theta(s_0) = {cal.atan_quad([ sh.y_s_dy_ds(s_0)[0][0] - rmsh.parameters["c"][0], sh.y_s_dy_ds(s_0)[0][1] - rmsh.parameters["c"][1] ])}')
# 


'''
return the parameter `s` of the curve `y(s)` corresponding to a point `y` on the plane
Input values: 
    - `y`: the point in the reference configuration
Return values: 
    - `s`: the value of the parametric coordinate of the curve `y(s)` such that the point y(s) forms the same polar angle as `y` with respect to `c`
'''
def s_y(y):

    # the polar angle that `y` forms with respect to `c``, theta_y is in [0, 2 pi]
    theta_y = cal.atan_quad([ y[0] - rmsh.parameters['c'][0], y[1] - rmsh.parameters['c'][1] ])

    print(f's_y has been called:\n \t theta_y = {theta_y}', flush=True)

    if 0 <= theta_y < theta_0:
        '''
            0 <= theta_y < theta_0: the solution s must lie in the interval s_0 <= s < 1.0. 
            Given that in this interval the angle 

                cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])

            may abruptly jump from 2 pi to 0, here I use the "bare" version of atan_quad 
            
                np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0]))

            that is continuous in the neighborhood of s ~ s_0. 
        '''

        print(f'Case I\ns_0 = {s_0}', flush=True)

        s = brentq(\
                lambda s: np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])) - theta_y, \
                a=s_0, b=1 \
            )

    else:
        # theta_0 <= theta_y < 2 pi: the solution s must lie in the interval 0 <= s < s_0

        print(f'Case II', flush=True)

        s = brentq(\
            lambda s: cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ]) - theta_y, 
            a=0, b=s_0 \
            )
        
    return s





class f_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of the parametric coordinate `s` corresponding to `x`
        s = s_y(x)

        values[0] = sh.y_s_dy_ds(s)[1][0]
        values[1] = sh.y_s_dy_ds(s)[1][1]

    def value_shape(self):
        return (2,)
    


    
class nu_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of `t` corresponding to `x`
        s = s_y(x)

        _, d_y_s_ds = sh.y_s_dy_ds(s)

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
