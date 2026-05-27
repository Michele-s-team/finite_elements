'''
this module solves for the fields v_n, sigma_n, u_n, u_dot_n which define the state of the whole system, for an elastic body in a fluid, where the fluid is contained into a closed box and both the elastic body and the fluid are subjected to gravity
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import continuation as cont
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])



# 1. define expressions for BCs

class v_lrtb_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
class sigma_t_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0

    def value_shape(self):
        return (1,)
    

class rho_el_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['rho_el']

    def value_shape(self):
        return (1,)
    
   
class f_ela_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = - rpam.parameters['rho_el'] * rpam.parameters['g']

    def value_shape(self):
        return (2,)
    

class f_fluid_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = - rpam.parameters['rho_fluid'] * rpam.parameters['g']

    def value_shape(self):
        return (2,)


msh.interpolate_dg(fsp.v_lrtb, v_lrtb_expression())

msh.interpolate_dg(fsp.sigma_t, sigma_t_expression())

msh.interpolate_dg(fsp.rho_el, rho_el_expression())

msh.interpolate_dg(fsp.f_ela, f_ela_expression())
msh.interpolate_dg(fsp.f_fluid, f_fluid_expression())





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
                                            - fsp.f_fluid[i] * fsp.nu_v_n[i]
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
                bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_n[i] * ela.detF(fsp.u_n) * rmsh.ds_mesh[0]['ds_r'] + \
                bgeo.facet_normal[0](sub_mesh_1_label)[l] * ela.G(fsp.u_n(sub_mesh_1_label))[l, j] * flu.sigma_ale(fsp.v_n(sub_mesh_1_label), fsp.sigma_n(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_n(sub_mesh_1_label)[i] * ela.detF(fsp.u_n(sub_mesh_1_label)) * rmsh.ds_mesh[0]['dS_shape']
           ) \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * ( \
            msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + (fsp.v_n[i] - fsp.v_lrtb[i]) * fsp.nu_v_n[i] * rmsh.ds_mesh[0]['ds'] \
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
        fsp.sigma_n * fsp.nu_sigma_n * rmsh.ds_mesh[0]['ds_t'] \
    )



# 2.2 elastic body and mesh

# 2.2.1 u_n


F_u_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        fsp.rho_el / dt * (fsp.u_dot_n[i] - fsp.u_dot_n_1[i]) * fsp.nu_u_n[i] \
                                        + ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * (fsp.nu_u_n[i].dx(k)) \
                                        - fsp.f_ela[i] * fsp.nu_u_n[i], 
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
            

F = F_v_n + F_sigma_n + F_u_n + F_u_dot_n