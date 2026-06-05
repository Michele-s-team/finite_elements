'''
this module solves for the fields v_n, sigma_n, u_n, u_dot_n which define the state of the whole system, for an elastic body in a channel in which fluid is injected in the left edge of the channel
'''

from fenics import *
import importlib
import numpy as np
from scipy.optimize import brentq
import ufl as ufl

import calculus as cal
import constants.utils as const
import continuation as cont
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)

i, j, k, l, m, n, o, p, q, r, s, t, u = ufl.indices(13)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])



'''# print facet_normal to check sub_mesh_0_label and sub_mesh_1_label
import input_output as io 
import solution_paths as solpath

n_1 = bgeo.field_facet_normal(bgeo.facet_normal[0](sub_mesh_1_label), rmsh.lmsh.mesh[0], rmsh.ds_mesh[0]['dS_shape'], interior=True)

io.full_print(n_1, 'n_1', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
'''



# 
def theta(s):
    return cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])




# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = theta(0)
print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# 1. Determine for s_0
# 1.1

if theta_0 == 0.0:
    # here s_0 = 1

    print(f'Setting s_0 = 1')

    s_0 = 1

else:
    # theta_0 != 0 -> need to solve for s_0

    print(f'Solving for s_0 ... ')

    # 1.1 find two values of s, s_0_L and s_0_R, that bracket s_0, by looking for a pair of values of s, s_i and s_i_plus_1 such that theta(s_i_plus_1) < theta(s_i). This is done by dividing the interval 0 < s < 1 into ~ N_scan interals. Note that N_scan needs to be large enough to resolve the root. 
    for i_s in range(rpam.parameters['N_scan']+1):
        
        if theta((i_s+1)/(rpam.parameters['N_scan']-1)) < theta(i_s/(rpam.parameters['N_scan']-1)): 
            break

    s_0_L = i_s/(rpam.parameters["N_scan"]-1)
    s_0_R = (i_s+1)/(rpam.parameters["N_scan"]-1)

    print(f's_0 lies betwee {s_0_L} and {s_0_R}')

    # 1.2 solve for s_0 by using s_0_L and s_0_R
    s_0 = brentq(lambda s:  np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=s_0_L, b=s_0_R)

    print(f's_0 = {s_0}\t theta(s_0) = {theta(s_0)}')

    print(f'... done. ')
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
        # theta_y takes the special values 0 or 2 pi -> set directly s = s_0

        s = s_0

    else:
        # theta_y is not equal to 0 nor to 2 pi -> solve for s

        if 0 < theta_y <= theta_0:
            
            # 0 < theta_y <= theta_0: the solution s must lie in the interval s_0 < s <= 1.0. 
            

            print(f'Case I\n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(s_0+const.epsilon)}\n\ttheta_R = {theta(1)}', flush=True)

            s = brentq(\
                    lambda s: theta(s) - theta_y, \
                    a=s_0 + const.epsilon, b=1 + const.epsilon \
                )

        else:
            # theta_0 <= theta_y < 2 pi,  the solution s must lie in the interval 0 <= s < s_0


            print(f'Case II \n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(0)}\n\ttheta_R = {theta(s_0)}', flush=True)

            s = brentq(\
                lambda s: theta(s) - theta_y, 
                a=0, b=s_0 - const.epsilon \
            )
 
    print(f'\t s = {s}')

    return s


# define expressions for  curvature computation


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

msh.interpolate_dg(fsp.f, f_expression())
msh.interpolate_dg(fsp.nu, nu_expression())

# 1. define expressions for BCs

class v_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['v_l'] * 4.0 * 1.5 * x[1] * (rmsh.parameters['h'] - x[1]) / (rmsh.parameters['h']**2)
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
class v_tb_circle_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
        

class rho_el_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['rho_el']

    def value_shape(self):
        return (1,)
    

msh.interpolate_dg(fsp.v_l, v_l_expression())
msh.interpolate_dg(fsp.v_tb, v_tb_circle_expression())


msh.interpolate_dg(fsp.rho_el, rho_el_expression())




bcs = []

# 2 variational problems

# 2.1 fluid

# 2.1.1 v_n

# natural BC imposed here
F_v_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.v_n[i] * fsp.nu_v_n[i], 
                                        ( \
                                            rpam.parameters['rho_fluid'] * ( (fsp.v_n[i] - fsp.v_n_1[i]) / dt \
                                            + (fsp.v_n[k] - fsp.u_dot_n[k]) * ela.G(fsp.u_n)[j, k] * (fsp.v_n[i]).dx(j) ) * fsp.nu_v_n[i] \
                                            + ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * (fsp.nu_v_n[i]).dx(k) \
                                        ) * ela.detF(fsp.u_n), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        - (\
            msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] ) \
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        - ( \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_l'] + \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_tb'] + \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, 1] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 1] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_r'] + \
                bgeo.facet_normal[0](sub_mesh_1_label)[l] * ela.G(fsp.u_n(sub_mesh_1_label))[l, j] * flu.sigma_ale(fsp.v_n(sub_mesh_1_label), fsp.sigma_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_n(sub_mesh_1_label)[i] * ela.detF(fsp.u_n(sub_mesh_1_label)) * rmsh.ds_mesh[0]['dS_shape']
           ) \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + (fsp.v_n[i] - fsp.v_l[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_l'] \
            + (fsp.v_n[i] - fsp.v_tb[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_tb'] \
        ) \
        + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * (\
             (fsp.v_n(sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i])) * fsp.nu_v_n(sub_mesh_1_label)[i] * rmsh.ds_mesh[0]['dS_shape']
         )



F_sigma_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.sigma_n * fsp.nu_sigma_n, 
                                        ela.G(fsp.u_n)[j, i] * fsp.v_n[i].dx(j) * fsp.nu_sigma_n * ela.detF(fsp.u_n),
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                    )  * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_sigma_n, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_I_square'] + \
        fsp.sigma_n * fsp.nu_sigma_n * rmsh.ds_mesh[0]['ds_r'] \
    )



# 2.2 elastic body and mesh

# 2.2.1 u_n


F_u_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.rho_el / dt * (fsp.u_dot_n[i] - fsp.u_dot_n_1[i]) * fsp.nu_u_n[i] \
                                        + ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * (fsp.nu_u_n[i].dx(k)), 
                                        - ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_n[k].dx(i)), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        - (\
                msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[k] * msh.average( ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] )
        ) * rmsh.ds_mesh[0]['dS_I_shape'] \
        - (flu.sigma_ale(fsp.v_n(sub_mesh_1_label), cont.pressure_scale * fsp.sigma_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j]) * bgeo.facet_normal[0](sub_mesh_0_label)[k]) * fsp.nu_u_n(sub_mesh_0_label)[i] * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_shape'] \
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
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_shape'] \
        ) \



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
                                        (fsp.u_n[i] - fsp.u_n_1[i] - fsp.u_dot_n[i] * dt) * fsp.nu_u_dot_n[i], 
                                        - Q(fsp.u_n, fsp.u_dot_n)[k, i] * (fsp.nu_u_dot_n[k]).dx(i), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
            + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_shape']
            ) \
            + ( msh.jump(fsp.nu_u_dot_n[k], bgeo.facet_normal[0])[i] * msh.average( Q(fsp.u_n, fsp.u_dot_n)[k, i] ) ) * rmsh.ds_mesh[0]['dS_I_square'] \
            + ( bgeo.facet_normal[0][i] * Q(fsp.u_n, fsp.u_dot_n)[k, i] * fsp.nu_u_dot_n[k] ) * rmsh.ds_mesh[0]['ds'] \
            + ( bgeo.facet_normal[0](sub_mesh_1_label)[i] * Q(fsp.u_n(sub_mesh_1_label), fsp.u_dot_n(sub_mesh_1_label))[k, i] * (fsp.nu_u_dot_n(sub_mesh_1_label))[k]) * rmsh.ds_mesh[0]['dS_shape'] \
            + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
                + ( fsp.u_dot_n[i] * fsp.nu_u_dot_n[i] ) * rmsh.ds_mesh[0]['ds'] \
            ) \
            + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] *  rmsh.ds_mesh[0]['dS_shape'] \
            ) \
            

# 2.3 variational problems for curvature computation

F_mu = (\
        (fsp.mu_n \
        - (fsp.f[i] + fsp.grad_u_n[i, k] * fsp.f[k]).dx(j) * fsp.f[j] \
        * ( sqrt( dot(fsp.f, fsp.f) / (ela.F(fsp.u_n)[p, q] * ela.F(fsp.u_n)[p, r] * fsp.f[q] * fsp.f[r] ) ) \
        * bgeo.epsilon[i, s] * ela.F(fsp.u_n)[s, t] * bgeo.epsilon[t, u] * fsp.nu[u] )  \
        / (2.0 * (fsp.f[m] + fsp.grad_u_n[m, n] * fsp.f[n]) * (fsp.f[m] + fsp.grad_u_n[m, o] * fsp.f[o]) ) \
        ) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.mu_n, bgeo.facet_normal[0])[i] *  msh.jump(fsp.nu_mu, bgeo.facet_normal[0])[i] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )

F_grad_u = ( (fsp.grad_u_n[i, j] - fsp.u_n[i].dx(j)) * fsp.nu_grad_u[i, j] ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.grad_u_n[i, j], bgeo.facet_normal[0])[k] *  msh.jump(fsp.nu_grad_u[i, j], bgeo.facet_normal[0])[k] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )


F = F_v_n + F_sigma_n + F_u_n + F_u_dot_n + F_mu + F_grad_u