'''
this module solves for the fields v_n, sigma_n_12, u_n, u_dot_n which define the state of the whole system
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh, rmsh.sf, rmsh.lmsh.parameters["sub_mesh_0_id"], rmsh.lmsh.parameters["sub_mesh_1_id"], rmsh.dS_ellipse)


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
    
msh.interpolate_dg(fsp.v_l, v_l_expression())
msh.interpolate_dg(fsp.v_tb, v_tb_circle_expression())


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
            + fsp.phi * fsp.nu_phi * rmsh.dS_ellipse
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
        ) * rmsh.dx

#sign
