'''
this module solves for the fields v_n, sigma_n_12, u_n, u_dot_n which define the state of the whole system
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import calculus as cal
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh, rmsh.sf, rmsh.lmsh.parameters["sub_mesh_0_id"], rmsh.lmsh.parameters["sub_mesh_1_id"], rmsh.dS_ellipse)



'''# print facet_normal to check sub_mesh_0_label and sub_mesh_1_label
import input_output as io 
import solution_paths as solpath

n_1 = bgeo.field_facet_normal(bgeo.facet_normal(sub_mesh_1_label), rmsh.lmsh.mesh, rmsh.dS_ellipse, interior=True)

io.full_print(n_1, 'n_1', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
'''


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

# {d y_s / ds}_notes
class dyds_ellipse_expression(UserExpression):
    def eval(self, values, x):

        s = 1 / (2 * np.pi) * cal.atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s)[1]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)
    
class f_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
    
msh.interpolate_dg(fsp.v_l, v_l_expression())
msh.interpolate_dg(fsp.v_tb, v_tb_circle_expression())

msh.interpolate_dg(fsp.rho_el, rho_el_expression())

msh.interpolate_dg(fsp.dyds_ellipse, dyds_ellipse_expression())

msh.interpolate_dg(fsp.f, f_expression())


bcs = []

# 2 variational problems

# 2.1 fluid

# 2.1.1 v_

# natural BC imposed here
F_v_ = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh,
                                        rmsh.sf, 
                                        fsp.v_[i] * fsp.nu_v_[i], 
                                        ( \
                                            rpam.parameters['rho_fluid'] * ( (fsp.v_[i] - fsp.v_n_1[i]) / dt \
                                            + ( 3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_dot_n_1[k]) * ela.G(fsp.u_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_dot_n_2[k]) * ela.G(fsp.u_n_2)[j, k]) * (fsp.V[i]).dx(j) ) * fsp.nu_v_[i] \
                                            + ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * (fsp.nu_v_[i]).dx(k) \
                                        ) * ela.detF(fsp.u_n), 
                                        rmsh.lmsh.parameters['sub_mesh_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_1_id']
                                ) * rmsh.dx \
        - (\
            msh.jump(fsp.nu_v_[i], bgeo.facet_normal)[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] ) \
            ) * rmsh.dS_I[1] \
        - ( \
                bgeo.facet_normal[l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_[i] * ela.detF(fsp.u_n) * rmsh.ds_l + \
                bgeo.facet_normal[l] * ela.G(fsp.u_n)[l, j] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n, rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_[i] * ela.detF(fsp.u_n) * rmsh.ds_tb + \
                bgeo.facet_normal[l] * ela.G(fsp.u_n)[l, 1] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n, rpam.parameters['mu_fluid'])[i, 1] * fsp.nu_v_[i] * ela.detF(fsp.u_n) * rmsh.ds_r + \
                bgeo.facet_normal(sub_mesh_1_label)[l] * ela.G(fsp.u_n(sub_mesh_1_label))[l, j] * flu.sigma_ale(fsp.V(sub_mesh_1_label), fsp.sigma_n_32(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * fsp.nu_v_(sub_mesh_1_label)[i] * ela.detF(fsp.u_n(sub_mesh_1_label)) * rmsh.dS_ellipse
           ) \
        + rpam.parameters['alpha']/rmsh.r_mesh * ( \
            msh.jump(fsp.v_[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_v_[i], bgeo.facet_normal)[j] * rmsh.dS_I[1] \
            + (fsp.v_[i] - fsp.v_l[i]) * fsp.nu_v_[i] * rmsh.ds_l \
            + (fsp.v_[i] - fsp.v_tb[i]) * fsp.nu_v_[i] * rmsh.ds_tb \
            + (fsp.v_(sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i])) * fsp.nu_v_(sub_mesh_1_label)[i] * rmsh.dS_ellipse
        )





# 2.1.2 phi

# natural BC imposed here
F_phi = msh.ufl_conditional_form(
            rmsh.lmsh.mesh,
            rmsh.sf, 
            fsp.phi * fsp.nu_phi, 
            ( \
                ela.G(fsp.u_n)[k, i] * ((fsp.v_[i]).dx(k)) * fsp.nu_phi \
                + dt / rpam.parameters['rho_fluid'] * ela.G(fsp.u_n)[k, i] * ela.G(fsp.u_n)[j, i] * (fsp.phi.dx(j)) * (fsp.nu_phi.dx(k)) \
            ) * ela.detF(fsp.u_n),
            rmsh.lmsh.parameters['sub_mesh_0_id'],
            rmsh.lmsh.parameters['sub_mesh_1_id']
        ) * rmsh.dx \
        - dt / rpam.parameters['rho_fluid'] * ( \
            ( msh.jump(fsp.nu_phi, bgeo.facet_normal)[k] * msh.average( ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, i] * ela.G(fsp.u_n)[j, i] * fsp.phi.dx(j) ) ) * rmsh.dS_I[1] \
            + ( bgeo.facet_normal[k]  * ela.G(fsp.u_n)[k, i] * ela.G(fsp.u_n)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi ) * ela.detF(fsp.u_n) * rmsh.ds_r \
        ) \
        + rpam.parameters['alpha']/rmsh.r_mesh * (\
            msh.jump(fsp.phi, bgeo.facet_normal)[i] * msh.jump(fsp.nu_phi, bgeo.facet_normal)[i] * rmsh.dS_I[1] \
            + fsp.phi * fsp.nu_phi * rmsh.ds_r
        )



# 2.1.3 v_n

F_v_n = msh.ufl_conditional_form(
            rmsh.lmsh.mesh,
            rmsh.sf, 
            fsp.v_n[i] * fsp.nu_v_n[i], 
            ( \
                ( (fsp.v_[i] - fsp.v_n[i]) - (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_n)[j, i] * (fsp.phi.dx(j)) ) * fsp.nu_v_n[i] \
            ) * ela.detF(fsp.u_n), 
            rmsh.lmsh.parameters['sub_mesh_0_id'],
            rmsh.lmsh.parameters['sub_mesh_1_id']
        ) * rmsh.dx \
        + rpam.parameters['alpha']/rmsh.r_mesh * ( \
            msh.jump(fsp.v_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_v_n[i], bgeo.facet_normal)[j] * rmsh.dS_I[1]
        )

#check


# 2.2 elastic body and mesh

# 2.2.1 u_n

F_u_n = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh,
                                        rmsh.sf, 
                                        (fsp.u_n[i] - fsp.u_n_1[i] - fsp.u_dot_n[i] * dt) * fsp.nu_u_dot_n[i], 
                                        - ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_n[k].dx(i)), 
                                        rmsh.lmsh.parameters['sub_mesh_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_1_id']
                                ) * rmsh.dx \
        + rpam.parameters['alpha']/rmsh.r_mesh * ( \
            msh.jump(fsp.u_dot_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal)[j] * rmsh.dS_I[0]
        ) \
        + (\
            msh.jump(fsp.nu_u_n[k], bgeo.facet_normal)[i] * msh.average( ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] )   
        ) * rmsh.dS_I[1] \
        + bgeo.facet_normal[i] * ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[k, i] * fsp.nu_u_n[k] * rmsh.ds_lrtb \
        + bgeo.facet_normal(sub_mesh_1_label)[i] * ela.P(fsp.u_n(sub_mesh_1_label), ela.K(fsp.u_n(sub_mesh_1_label), rpam.parameters['exponent']), ela.mu(fsp.u_n(sub_mesh_1_label), rpam.parameters['exponent']))[k, i] * fsp.nu_u_n(sub_mesh_1_label)[k] * rmsh.dS_ellipse \
        + rpam.parameters['alpha']/rmsh.r_mesh * (\
            msh.jump(fsp.u_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal)[j] * (rmsh.dS_I[1] + rmsh.dS_ellipse) \
            + fsp.u_n[i] * fsp.nu_u_n[i] * rmsh.ds_lrtb \
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
                                        rmsh.lmsh.mesh,
                                        rmsh.sf, 
                                        fsp.rho_el / dt * (fsp.u_dot_n[i] - fsp.u_dot_n_1[i]) * fsp.nu_u_n[i] \
                                        + ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * (fsp.nu_u_n[i].dx(k)), 
                                        - Q(fsp.u_n, fsp.u_dot_n)[k, i] * (fsp.nu_u_dot_n[k]).dx(i), 
                                        rmsh.lmsh.parameters['sub_mesh_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_1_id']
                                ) * rmsh.dx \
            - (\
                msh.jump(fsp.nu_u_n[i], bgeo.facet_normal)[k] * msh.average( ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] )
            ) * rmsh.dS_I[0] \
            - bgeo.facet_normal[k] * ela.N(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * fsp.nu_u_n[i] * rmsh.ds_circle \
            - (flu.sigma_ale_no_pressure(fsp.v_n(sub_mesh_1_label), fsp.sigma_n_12(sub_mesh_1_label), fsp.u_n(sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[j, k]) * bgeo.facet_normal(sub_mesh_0_label)[j]) * fsp.nu_u_n(sub_mesh_0_label)[i] * rmsh.dS_ellipse \
            + rpam.parameters['alpha']/rmsh.r_mesh * ( \
                msh.jump(fsp.u_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_u_n[i], bgeo.facet_normal)[j] * rmsh.dS_I[0] \
                + fsp.u_n[i] * fsp.nu_u_n[i] * rmsh.ds_circle
            ) \
            + ( msh.jump(fsp.nu_u_dot_n[k], bgeo.facet_normal)[i] * msh.average( Q(fsp.u_n, fsp.u_dot_n)[k, i] ) ) * rmsh.dS_I[1] \
            + ( bgeo.facet_normal[i] * Q(fsp.u_n, fsp.u_dot_n)[k, i] * fsp.nu_u_dot_n[k] ) * rmsh.ds_lrtb \
            + ( bgeo.facet_normal(sub_mesh_1_label)[i] * Q(fsp.u_n(sub_mesh_1_label), fsp.u_dot_n(sub_mesh_1_label))[k, i] * (fsp.nu_u_dot_n(sub_mesh_1_label))[k]) * rmsh.dS_ellipse \
            + rpam.parameters['alpha']/rmsh.r_mesh * ( \
                msh.jump(fsp.u_dot_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.nu_u_dot_n[i], bgeo.facet_normal)[j] * (rmsh.dS_I[1] + rmsh.dS_ellipse) \
                + ( fsp.u_dot_n[i] * fsp.nu_u_dot_n[i] ) * rmsh.ds_lrtb \
            )
            


F = F_v_ + F_phi + F_v_n + F_u_n + F_u_dot_n