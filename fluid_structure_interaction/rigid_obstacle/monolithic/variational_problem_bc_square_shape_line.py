'''
this module solves for the fields theta_n, omega_n, v_n, sigma_n, u_n, u_dot_n which define the state of the whole system, for a rigid body in a channel in which fluid is injected in the left edge of the channel
'''

from fenics import *
import importlib
import sys
import ufl as ufl

import calculus as cal
import differential_geometry.boundary.geometry as bgeo
import function as fu
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)


i, j, k, l, m = ufl.indices(5)


'''
rotation matrix as tensor

Input values: 
    - `theta`: the rotation angle (in radians)
Return values: 
    - R(theta) expresssed as a 2 x 2 ufl tensor
'''

def R(theta):

    return as_tensor([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

'''
derivative of the rotation matrix as tensor

Input values: 
    - `theta`: the rotation angle (in radians)
Return values: 
    - dR(theta)/dtheta expresssed as a 2 x 2 ufl tensor
'''

def dRdtheta(theta):

    return as_tensor([[-sin(theta), -cos(theta)], [cos(theta), -sin(theta)]])


def c():
    return as_vector(rmsh.parameters['c'])


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size


# 1.  expressions for BCs
# `v_l` and `v_tb` correspond to `g` in `Channel flow with moving obstacle`

class v_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['v_l'] * 4.0 * 1.5 * x[1] * (rmsh.parameters['h'] - x[1]) / (rmsh.parameters['h']**2)
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
class v_tb_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


        
msh.interpolate_dg(fsp.v_l, v_l_expression())
msh.interpolate_dg(fsp.v_tb, v_tb_expression())
msh.interpolate_dg(fsp.y, fu.identity_expression())


bcs = []


# 2. variational problems

# 2.1 fluid

# 2.1.1 v_n

# natural BC imposed here
F_v_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.v_n[i] * fsp.nu_v_n[i] * ela.detF(fsp.u_n), 
                                        ( \
                                            rpam.parameters['rho'] * ( (fsp.v_n[i] - fsp.v_n_1[i]) / dt \
                                            + (fsp.v_n[k] - fsp.u_dot_n[k]) * ela.G(fsp.u_n)[j, k] * (fsp.v_n[i]).dx(j) ) * fsp.nu_v_n[i] \
                                            + ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, j] * (fsp.nu_v_n[i]).dx(k) \
                                        ) * ela.detF(fsp.u_n), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        - (\
            msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, j] ) \
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        - ( \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_l'] + \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_tb'] + \
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, 1] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, 1] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_r'] + \
                msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_1)[l] * ela.G(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1))[l, j] * flu.sigma_ale(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu'])[i, j] * msh.side(fsp.nu_v_n, rmsh.I_sub_mesh_0_1)[i] * ela.detF(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1)) * rmsh.ds_mesh[0]['dS_shape']
           ) \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + (fsp.v_n[i] - fsp.v_l[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_l'] \
            + (fsp.v_n[i] - fsp.v_tb[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds_tb'] \
        ) \
        + rpam.parameters['alpha_shape']/rmsh.r_mesh[0] * (\
             (
                msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1)[i] \
                - ( msh.average(fsp.omega_n) * dRdtheta(msh.average(fsp.theta_n - Constant(rmsh.parameters['phi'])))[i, j] * ( msh.average(fsp.y)[j] - c()[j] ) )
            ) * msh.side(fsp.nu_v_n, rmsh.I_sub_mesh_0_1)[i] * rmsh.ds_mesh[0]['dS_shape']
         )




F_sigma_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.sigma_n * fsp.nu_sigma_n * ela.detF(fsp.u_n), 
                                        ela.G(fsp.u_n)[j, i] * fsp.v_n[i].dx(j) * fsp.nu_sigma_n * ela.detF(fsp.u_n),
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                    )  * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_sigma_n, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_I_square'] + \
        fsp.sigma_n * fsp.nu_sigma_n * rmsh.ds_mesh[0]['ds_r'] \
    )




# 2.2 elastic body and domain

# 2.2.1 u_n


F_u_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        ( fsp.u_n[i] - ( c()[i] - msh.average(fsp.y)[i] + R(fsp.theta_n - Constant(rmsh.parameters['phi']))[i, j] * ( msh.average(fsp.y)[j] - c()[j] ) ) ) * fsp.nu_u_n[i], 
                                        - ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_n[k].dx(i)), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        + (\
            msh.jump(fsp.nu_u_n[k], bgeo.facet_normal[0])[i] * msh.average( ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] )   
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        + bgeo.facet_normal[0][i] * ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * fsp.nu_u_n[k] * rmsh.ds_mesh[0]['ds'] \
        + msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_1)[i] * ela.P(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), ela.K(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.mu(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']))[k, i] * msh.side(fsp.nu_u_n, rmsh.I_sub_mesh_0_1)[k] * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + fsp.u_n[i] * fsp.nu_u_n[i] * rmsh.ds_mesh[0]['ds'] \
        ) \
        + rpam.parameters['alpha_shape']/rmsh.r_mesh[0] * (\
            msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_shape'] \
        ) \



# 2.2.2 u_dot_n

'''
ela.Q(fsp.u_n, fsp.u_dot_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']), ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']), ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']))

ela.Q(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), ela.K(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.mu(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.K_dot(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.mu_dot(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']))
'''

F_u_dot_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        (fsp.u_dot_n[i] - ( fsp.omega_n * dRdtheta(fsp.theta_n - Constant(rmsh.parameters['phi']))[i, j] * ( (fsp.y)[j] - c()[j] ) ) ) * fsp.nu_u_dot_n[i], 
                                        - ela.Q(fsp.u_n, fsp.u_dot_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']), ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']), ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_dot_n[k]).dx(i), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
            + ( msh.jump(fsp.nu_u_dot_n[k], bgeo.facet_normal[0])[i] * msh.average( ela.Q(fsp.u_n, fsp.u_dot_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']), ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']), ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']))[k, i] ) ) * rmsh.ds_mesh[0]['dS_I_square'] \
            + ( bgeo.facet_normal[0][i] * ela.Q(fsp.u_n, fsp.u_dot_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']), ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']), ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']))[k, i] * fsp.nu_u_dot_n[k] ) * rmsh.ds_mesh[0]['ds'] \
            + ( msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_1)[i] * ela.Q(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), ela.K(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.mu(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.K_dot(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']), ela.mu_dot(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1),  msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1), rpam.parameters['exponent']))[k, i] * (msh.side(fsp.nu_u_dot_n, rmsh.I_sub_mesh_0_1))[k]) * rmsh.ds_mesh[0]['dS_shape'] \
            + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
                + ( fsp.u_dot_n[i] * fsp.nu_u_dot_n[i] ) * rmsh.ds_mesh[0]['ds'] \
            ) \
            + rpam.parameters['alpha_shape']/rmsh.r_mesh[0] * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal[0])[j] *  rmsh.ds_mesh[0]['dS_shape'] \
            ) \
            

# 2.3 theta_n and omega_n

# 2.3.1 theta_n

F_theta_n = ( ( fsp.theta_n - ( fsp.theta_n_1 + fsp.omega_n_1 * dt ) ) * fsp.nu_omega_n ) * rmsh.dx_mesh[0]['dx']

# sign
sys.exit(1)


# 2.3.2 omega_n

F = F_v_n + F_sigma_n + F_u_n + F_u_dot_n