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
def theta(s):
    return cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])


# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = theta(0)
print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# solve for s_0
# REVISE: INCLUDE A SYSTEMATIC WAY TO DETERMINE THE INTERVAL WHERE TO LOOK FOR S_0
print(f'Solving for s_0 ... ')

print(f'\n\tf_L = {np.arctan((sh.y_s_dy_ds(0)[0][1] - rmsh.parameters["c"][1]) / (sh.y_s_dy_ds(0)[0][0] - rmsh.parameters["c"][0]))}')
print(f'\n\tf_R = {np.arctan((sh.y_s_dy_ds(0.25)[0][1] - rmsh.parameters["c"][1]) / (sh.y_s_dy_ds(0.25)[0][0] - rmsh.parameters["c"][0]))}')
s_0 = brentq(lambda s:  np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=0.0, b=0.25 )


print(f'... done. ')
print(f's_0 = {s_0}\t theta(s_0) = {theta(s_0)}')
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

    if (abs(theta_y) < const.epsilon) or (abs(theta_y - 2.0 * np.pi) < const.epsilon): 

        s = s_0

    else:

        if 0 < theta_y <= theta_0:
            '''
                0 < theta_y <= theta_0: the solution s must lie in the interval s_0 < s <= 1.0. 
            '''

            print(f'Case I\n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(s_0+const.epsilon)}\n\ttheta_R = {theta(1)}', flush=True)

            s = brentq(\
                    lambda s: theta(s) - theta_y, \
                    a=s_0 + const.epsilon, b=1 + const.epsilon \
                )

        else:
            # theta_0 <= theta_y < 2 pi

            if theta_0 != 0.0:
                #  the solution s must lie in the interval 0 <= s < s_0


                print(f'Case IIa \n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(0)}\n\ttheta_R = {theta(s_0)}', flush=True)

                s = brentq(\
                    lambda s: theta(s) - theta_y, 
                    a=0, b=s_0 - const.epsilon \
                )
            else:

                #  the solution s must lie in the interval 0 <= s < 1


                print(f'Case IIb \n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(0)}\n\ttheta_R = {theta(1 - const.epsilon)}', flush=True)

                s = brentq(\
                    lambda s: theta(s) - theta_y, 
                    a=0, b=1 - const.epsilon \
                )


        
    print(f'\t s = {s}')
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
