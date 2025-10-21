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

i, j, k, l = ufl.indices(4)

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

# sign

# step 1 for v_
F_v_fl_bar = ( \
                   rpam.parameters['rho_fl'] * ((fsp.v_fl_bar[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_dot_n_1[k]) * ela.G(fsp.u_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_dot_n_2[k]) * ela.G(fsp.u_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_v_[i]).dx(l) + rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[k, j] * ((fsp.V[i]).dx(k)) * ela.G(fsp.u_n_1)[l, j] * (fsp.nu_v_[i]).dx(l) \
           ) * ela.detF(fsp.u_n_1) * rmsh.dx \
       - (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * fsp.sigma_n_32 * fsp.nu_v_[i]) * ela.detF(fsp.u_n_1) * rmsh.ds \
       - ( \
                   rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_l \
                   + rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_tb \
                   + rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_ellipse \
                   + rpam.parameters['eta_fl'] * ela.G(fsp.u_n_1)[l, 1] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, 1] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_r \
           )

'''



# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.parameters['rho_fl'] / dt) * ela.G(fsp.u_n_1)[j, i] * ((fsp.v_fl_bar[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx \
        + (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_n_1) * rmsh.ds_r

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_fl_bar[i]) + (dt / rpam.parameters['rho_fl']) * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l))) * fsp.nu_v_n[i]) * ela.detF(fsp.u_n_1) * rmsh.dx
'''