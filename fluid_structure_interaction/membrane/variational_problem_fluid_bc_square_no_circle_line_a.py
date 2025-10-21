'''
this module solves for the fields, \textrm_{v_FL}^n, \varsigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta, beta, gamma, beta = ufl.indices(7)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


class v_fl_bar_b_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = rpam.parameters['v_fl_bar_b_const']* 4.0 * 1.5 * x[0] * (rmsh.parameters['L'] - x[0]) / (rmsh.parameters['L']**2)

    def value_shape(self):
        return (2,)


fsp.v_fl_bar_b.interpolate(v_fl_bar_b_Expression(element=fsp.Q_v_fl_bar.ufl_element()))

# BCs
# 1) for step 1
bc_v_fl_bar_b = DirichletBC(fsp.Q_v_fl_bar, fsp.v_fl_bar_b, rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_b_id"])
bc_v_fl_bar_l = DirichletBC(fsp.Q_v_fl_bar, Constant((0, 0)), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_l_id"])
bc_v_fl_bar_0_r = DirichletBC(fsp.Q_v_fl_bar.sub(0), Constant(0), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_r_id"])
bc_v_fl_bar_t = DirichletBC(fsp.Q_v_fl_bar, fsp.u_dot_n, rmsh.mf_sub_mesh[0], rmsh.parameters["sub_mesh_1_id"])

bc_v_fl_bar = [bc_v_fl_bar_b, bc_v_fl_bar_l, bc_v_fl_bar_0_r, bc_v_fl_bar_t]

# 2) for step 2
bc_phi_fl_b = DirichletBC(fsp.Q_phi_fl, Constant(0), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_b_id"])

bc_phi_fl = [bc_phi_fl_b]


# step 1 for v_
F_v_fl_bar = ( \
                   rpam.parameters['rho_fl'] * (
                                                (fsp.v_fl_bar[alpha] - fsp.v_fl_n_1[alpha]) / dt \
                                                + (3.0 / 2.0 * (fsp.v_fl_n_1[gamma] - fsp.u_dot_n_1[gamma]) * ela.G(fsp.u_n_1)[beta, gamma] - 1.0 / 2.0 * (fsp.v_fl_n_2[gamma] - fsp.u_dot_n_2[gamma]) * ela.G(fsp.u_n_2)[beta, gamma]) * (fsp.V_fl[alpha]).dx(beta)
                                                ) * fsp.nu_v_fl_bar[alpha] \
                    + fsp.sigma_fl_n_32 * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.nu_v_fl_bar[alpha]).dx(beta) \
                    + rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[gamma, beta] * ((fsp.V_fl[alpha]).dx(gamma)) * ela.G(fsp.u_n_1)[delta, beta] * (fsp.nu_v_fl_bar[alpha]).dx(delta) \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx_sub_mesh[0] \
            - (ela.G(fsp.u_n_1)[beta, alpha] * bgeo.facet_normal[beta] * fsp.sigma_fl_n_32 * fsp.nu_v_fl_bar[alpha]) * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds'] \
            - ( \
                   rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[delta, beta] * bgeo.facet_normal[delta] * ela.G(fsp.u_n_1)[gamma, beta] * (fsp.V_fl[alpha].dx(gamma)) * fsp.nu_v_fl_bar[alpha] * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_l'] + \
                   rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[delta, beta] * bgeo.facet_normal[delta] * ela.G(fsp.u_n_1)[gamma, beta] * (fsp.V_fl[alpha].dx(gamma)) * fsp.nu_v_fl_bar[alpha] * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_tb'] \
# sign
            )

'''



# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.parameters['rho_fl'] / dt) * ela.G(fsp.u_n_1)[j, i] * ((fsp.v_fl_bar[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx \
        + (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_n_1) * rmsh.ds_r

# step 3 for v_n
F_v_n = (((fsp.v_fl_n[i] - fsp.v_fl_bar[i]) + (dt / rpam.parameters['rho_fl']) * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l))) * fsp.nu_v_n[i]) * ela.detF(fsp.u_n_1) * rmsh.dx
'''