'''
this module solves for the fields v_n, sigma_n, u_n, u_dot_n, c_n which define the state of the whole system, for a fluid obstacle in a fluid channel in a closed box
'''

from fenics import *
import importlib
import numpy as np
from scipy.optimize import brentq
import ufl as ufl

import calculus as cal 
import constants.utils as const
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.diffusion as dif
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)


i, j, k, l, m = ufl.indices(5)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])

# dyds - start

def theta(s):
    return cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])



# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = theta(0)
# print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# 1. Determine  s_0
# 1.1

if (abs(theta_0) < const.epsilon) or (abs(theta_0 - 2*np.pi) < const.epsilon):
    # here s_0 = 1

    theta_0 = 0

    # print(f'** Setting s_0 = 1')

    s_0 = 1

else:
    # theta_0 != 0 -> need to solve for s_0

    # print(f'** Solving for s_0 ... ')

    # 1.1 find two values of s, s_0_L and s_0_R, that bracket s_0, by looking for a pair of values of s, s_i and s_i_plus_1 such that theta(s_i_plus_1) < theta(s_i). This is done by dividing the interval 0 < s < 1 into ~ N_scan interals. Note that N_scan needs to be large enough to resolve the root. 
    for i_s in range(rpam.parameters['N_scan']+1):
        
        if theta((i_s+1)/(rpam.parameters['N_scan']-1)) < theta(i_s/(rpam.parameters['N_scan']-1)): 
            break

    s_0_L = i_s/(rpam.parameters["N_scan"]-1)
    s_0_R = (i_s+1)/(rpam.parameters["N_scan"]-1)

    # print(f's_0 lies betwee {s_0_L} and {s_0_R}')

    # 1.2 solve for s_0 by using s_0_L and s_0_R
    s_0 = brentq(lambda s:  np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=s_0_L, b=s_0_R)

    # print(f's_0 = {s_0}\t theta(s_0) = {theta(s_0)}')

    # print(f'... done. ')
# 


# return the parameter `s` of the curve `y(s)` corresponding to a point `y` on the plane
# Input values: 
#     - `y`: the point in the reference configuration
# Return values: 
#     - `s`: the value of the parametric coordinate of the curve `y(s)` such that the point y(s) forms the same polar angle as `y` with respect to `c`

def s_y(y):

    # the polar angle that `y` forms with respect to `c``, theta_y is in [0, 2 pi]
    theta_y = cal.atan_quad([ y[0] - rmsh.parameters['c'][0], y[1] - rmsh.parameters['c'][1] ])

    if (abs(theta_y) < const.epsilon) or (abs(theta_y - 2.0 * np.pi) < const.epsilon): 
        # theta_y takes the special values 0 or 2 pi -> set directly s = s_0

        s = s_0

    else:
        # theta_y is not equal to 0 nor to 2 pi -> solve for s

        if 0 < theta_y <= theta_0:
            
            # 0 < theta_y <= theta_0: the solution s must lie in the interval s_0 < s <= 1.0. 
            

            s_L = s_0 + const.epsilon
            s_R = 1 + const.epsilon
            
            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)
                            
            # print(f'Case I\n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(s_0+const.epsilon)}\n\ttheta_R = {theta(1 + const.epsilon)}', flush=True)

        else:
            # theta_0 <= theta_y < 2 pi,  the solution s must lie in the interval 0 <= s < s_0

            if theta(0 + const.epsilon) < theta_y: 

                s_L = 0 + const.epsilon

            else:

                s_L = 0

            s_R = s_0 - const.epsilon

            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)

    # print(f'\t s = {s}')

    return s


class dyds_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of the parametric coordinate `s` corresponding to `x`
        s = s_y(x)

        values[0] = sh.y_s_dy_ds(s)[1][0]
        values[1] = sh.y_s_dy_ds(s)[1][1]

    def value_shape(self):
        return (2,)
    
# dyds - end


# 1. define expressions for BCs

class v_lrb_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
    
class f_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = - rpam.parameters['rho_shape'] * rpam.parameters['g']

    def value_shape(self):
        return (2,)


class f_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = - rpam.parameters['rho_square'] * rpam.parameters['g']

    def value_shape(self):
        return (2,)
    

class t_t_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    

class sigma_square_t_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['sigma_square_t']

    def value_shape(self):
        return (1,)


msh.interpolate_dg(fsp.v_lrb, v_lrb_expression())

msh.interpolate_dg(fsp.f_shape, f_shape_expression())
msh.interpolate_dg(fsp.f_square, f_square_expression())

msh.interpolate_dg(fsp.t_t, t_t_expression())
msh.interpolate_dg(fsp.sigma_square_t, sigma_square_t_expression())

'''
force per unit length exerted on the boundary of the shape fluid
Input values: 
    - `c`: concentration field
    - `u`: displacement field

Return values: 
    - \textrm{f}_alpha in fluid-structure interaction/fluid obstacle/notes
'''
def f_shape(c, u):

    return as_tensor([0, 0])

bcs = []

# 2 variational problems

# 2.1 fluid

# 2.1.1 v_n

# natural BC imposed here
F_v_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        ( \
                                            (rpam.parameters['rho_shape'] * ( (fsp.v_n[i] - fsp.v_n_1[i]) / dt \
                                            + (fsp.v_n[k] - fsp.u_dot_n[k]) * ela.G(fsp.u_n)[j, k] * (fsp.v_n[i]).dx(j) ) - fsp.f_shape[i] ) * fsp.nu_v_n[i] \
                                            + ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_shape'])[i, j] * (fsp.nu_v_n[i]).dx(k) \
                                        ) * ela.detF(fsp.u_n)
                                        ,
                                        ( \
                                            (rpam.parameters['rho_square'] * ( (fsp.v_n[i] - fsp.v_n_1[i]) / dt \
                                            + (fsp.v_n[k] - fsp.u_dot_n[k]) * ela.G(fsp.u_n)[j, k] * (fsp.v_n[i]).dx(j) ) - fsp.f_square[i] ) * fsp.nu_v_n[i] \
                                            + ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] * (fsp.nu_v_n[i]).dx(k) \
                                        ) * ela.detF(fsp.u_n), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        - (\
            msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_shape'])[i, j] ) \
        ) * rmsh.ds_mesh[0]['dS_I_shape'] \
        - ( \
                ( \
                    ela.detF(fsp.u_n(sub_mesh_0_label)) * bgeo.facet_normal[0](sub_mesh_0_label)[k] * ela.G(fsp.u_n(sub_mesh_1_label))[k, j] * flu.sigma_ale(fsp.v_n(sub_mesh_1_label), fsp.sigma_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_square'])[i, j] \
                    + f_shape(fsp.c_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label))[i] \
                )* fsp.nu_v_n(sub_mesh_0_label)[i]
        ) * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                    msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_shape'] \
                ) \
        - (\
            msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] ) \
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        - ( \
                bgeo.facet_normal[0](sub_mesh_1_label)[k] * ela.detF(fsp.u_n(sub_mesh_1_label)) * ela.G(fsp.u_n(sub_mesh_1_label))[k, j] * flu.sigma_ale(fsp.v_n(sub_mesh_1_label), fsp.sigma_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_square'])[i, j] * fsp.nu_v_n(sub_mesh_1_label)[i] * rmsh.ds_mesh[0]['dS_shape'] \
                + bgeo.facet_normal[0][k] * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_lr'] \
                + bgeo.facet_normal[0][k] * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_b'] \
                + fsp.t_t[i] * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_t'] \
           ) \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + (fsp.v_n[i] - fsp.v_lrb[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_lr'] \
            + (fsp.v_n[i] - fsp.v_lrb[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_b'] \
        ) \
        + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * (\
             msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_shape']
         )


F_sigma_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        ela.G(fsp.u_n)[j, i] * fsp.v_n[i].dx(j) * fsp.nu_sigma_n * ela.detF(fsp.u_n), 
                                        ela.G(fsp.u_n)[j, i] * fsp.v_n[i].dx(j) * fsp.nu_sigma_n * ela.detF(fsp.u_n),
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                    )  * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_sigma_n, bgeo.facet_normal[0])[i] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square']) \
        + (fsp.sigma_n - fsp.sigma_square_t) * fsp.nu_sigma_n * rmsh.ds_mesh[0]['ds_t'] \
    )




# 2.2 mesh

# 2.2.1 u_n

F_u_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        - ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_n[k].dx(i)), 
                                        - ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_n[k].dx(i)), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        + (\
            msh.jump(fsp.nu_u_n[k], bgeo.facet_normal[0])[i] * msh.average( ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] )  
        ) * rmsh.ds_mesh[0]['dS_I_shape'] \
        + bgeo.facet_normal[0](sub_mesh_0_label)[i] * ela.P(fsp.u_n(sub_mesh_0_label), ela.K(fsp.u_n(sub_mesh_0_label), rpam.parameters['exponent']), ela.mu(fsp.u_n(sub_mesh_0_label), rpam.parameters['exponent']))[k, i] * fsp.nu_u_n(sub_mesh_0_label)[k] * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_shape'] \
        ) \
        + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_shape'] \
        ) \
        + (\
            msh.jump(fsp.nu_u_n[k], bgeo.facet_normal[0])[i] * msh.average( ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] )  
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        + bgeo.facet_normal[0][i] * ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * fsp.nu_u_n[k] * rmsh.ds_mesh[0]['ds'] \
        + bgeo.facet_normal[0](sub_mesh_1_label)[i] * ela.P(fsp.u_n(sub_mesh_1_label), ela.K(fsp.u_n(sub_mesh_1_label), rpam.parameters['exponent']), ela.mu(fsp.u_n(sub_mesh_1_label), rpam.parameters['exponent']))[k, i] * fsp.nu_u_n(sub_mesh_1_label)[k] * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + fsp.u_n[i] * fsp.nu_u_n[i] * rmsh.ds_mesh[0]['ds'] \
        ) \
        + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * (\
            ( \
                ( \
                   fsp.u_n(sub_mesh_1_label)[i] \
                    - ( \
                        fsp.u_n_1(sub_mesh_1_label)[i] + ( fsp.v_n(sub_mesh_1_label)[j] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[j] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] * dt  \
                    ) \
                ) \
                * ( fsp.nu_u_n(sub_mesh_1_label)[i] \
                   - ( \
                        ( fsp.nu_v_n(sub_mesh_1_label)[k] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                        + ( fsp.v_n(sub_mesh_1_label)[k] * bgeo.delta_n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.nu_u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                        + ( fsp.v_n(sub_mesh_1_label)[k] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.delta_n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.nu_u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                     ) * dt \
                )
            ) * rmsh.ds_mesh[0]['dS_shape'] \
        )


# 2.2.2 u_dot_n

def Q(u, u_dot):

    return as_tensor(
        (ela.F_dot(u_dot)[k, j] * ela.S(u, ela.K(u, rpam.parameters['exponent']), ela.mu(u, rpam.parameters['exponent']))[j, i] \
        + ela.F(u)[k, j] * ela.S_dot(u,
                                    u_dot,
                                    ela.K(u, rpam.parameters['exponent']),
                                    ela.K_dot(u, u_dot, rpam.parameters['exponent']),
                                    ela.mu(u, rpam.parameters['exponent']),
                                    ela.mu_dot(u, u_dot, rpam.parameters['exponent']))[j, i]), 
    (k, i))


F_u_dot_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        - Q(fsp.u_n, fsp.u_dot_n)[k, i] * (fsp.nu_u_dot_n[k]).dx(i), 
                                        - Q(fsp.u_n, fsp.u_dot_n)[k, i] * (fsp.nu_u_dot_n[k]).dx(i), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
            + ( msh.jump(fsp.nu_u_dot_n[k], bgeo.facet_normal[0])[i] * msh.average( Q(fsp.u_n, fsp.u_dot_n)[k, i] ) ) * rmsh.ds_mesh[0]['dS_I_shape'] \
            + ( bgeo.facet_normal[0](sub_mesh_0_label)[i] * Q(fsp.u_n(sub_mesh_0_label), fsp.u_dot_n(sub_mesh_0_label))[k, i] * (fsp.nu_u_dot_n(sub_mesh_0_label))[k]) * rmsh.ds_mesh[0]['dS_shape'] \
            + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_shape']
            ) \
            + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] *  rmsh.ds_mesh[0]['dS_shape'] \
            ) \
            + ( msh.jump(fsp.nu_u_dot_n[k], bgeo.facet_normal[0])[i] * msh.average( Q(fsp.u_n, fsp.u_dot_n)[k, i] ) ) * rmsh.ds_mesh[0]['dS_I_square'] \
            + ( bgeo.facet_normal[0][i] * Q(fsp.u_n, fsp.u_dot_n)[k, i] * fsp.nu_u_dot_n[k] ) * rmsh.ds_mesh[0]['ds'] \
            + ( bgeo.facet_normal[0](sub_mesh_1_label)[i] * Q(fsp.u_n(sub_mesh_1_label), fsp.u_dot_n(sub_mesh_1_label))[k, i] * (fsp.nu_u_dot_n(sub_mesh_1_label))[k]) * rmsh.ds_mesh[0]['dS_shape'] \
            + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
                + ( fsp.u_dot_n[i] * fsp.nu_u_dot_n[i] ) * rmsh.ds_mesh[0]['ds'] \
            ) \
            + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * (\
            ( \
                ( \
                   fsp.u_dot_n(sub_mesh_1_label)[i] \
                    - ( fsp.v_n(sub_mesh_1_label)[j] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[j] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                ) \
                * ( fsp.nu_u_dot_n(sub_mesh_1_label)[i] \
                   - ( \
                        ( fsp.nu_v_n(sub_mesh_1_label)[k] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                        + ( fsp.v_n(sub_mesh_1_label)[k] * bgeo.delta_n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.nu_u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                        + ( fsp.v_n(sub_mesh_1_label)[k] * bgeo.n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[k] ) * bgeo.delta_n_cur(bgeo.facet_normal[0](sub_mesh_0_label), fsp.u_n(sub_mesh_1_label), fsp.nu_u_n(sub_mesh_1_label), fsp.dyds(sub_mesh_1_label))[i] \
                     ) \
                    )
                ) * rmsh.ds_mesh[0]['dS_shape'] \
            )
        

F_c_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.c_n * fsp.nu_c_n, 
                                        ( \
                                            ( (fsp.c_n - fsp.c_n_1)/dt  - ela.G(fsp.u_n)[i, j] * fsp.u_dot_n[j] * fsp.c_n.dx(i) ) * fsp.nu_c_n \
                                            - ela.G(fsp.u_n)[k, i] * dif.J_ale(fsp.u_n, fsp.c_n, fsp.v_n, rpam.parameters['D'])[i] * fsp.nu_c_n.dx(k)              
                                        ) * ela.detF(fsp.u_n), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        + (\
            msh.jump(fsp.nu_c_n, bgeo.facet_normal[0])[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, i] * dif.J_ale(fsp.u_n, fsp.c_n, fsp.v_n, rpam.parameters['D'])[i] ) \
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        + bgeo.facet_normal[0][k] * ela.G(fsp.u_n)[k, i] * fsp.v_n[i] * fsp.c_n * fsp.nu_c_n * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds'] \
        + ( \
            - rpam.parameters['kappa'] \
            + bgeo.facet_normal[0](sub_mesh_1_label)[k] * ela.detF(fsp.u_n(sub_mesh_1_label)) * ela.G(fsp.u_n(sub_mesh_1_label))[k, i] * fsp.v_n(sub_mesh_1_label)[i] * fsp.c_n(sub_mesh_1_label) \
        ) * fsp.nu_c_n(sub_mesh_1_label) * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.c_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_c_n, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_I_square'] \
        )
# sign

F = F_v_n + F_sigma_n + F_u_n + F_u_dot_n + F_c_n