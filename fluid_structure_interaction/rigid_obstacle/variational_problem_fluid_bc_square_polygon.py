'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.parameters["T"] / rpam.parameters["num_steps"]  # time step size

'''
this is the RHS of (63) of 'Channel flow with moving obstacle'
'''
class v__profile_l_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters["v_l"] * 4.0*1.5*x[1]*(rmsh.parameters["h"] - x[1]) / (rmsh.parameters["h"]**2)
        values[1] = 0
  
    def value_shape(self):
        return (2,)

'''
this is the RHS of (64) of 'Channel flow with moving obstacle'
'''
class v__profile_ellipse_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = fsp.omega_n * (-np.sin(fsp.theta_n) * (x[0] - rmsh.parameters["c"][0]) - np.cos(fsp.theta_n) * (x[1] - rmsh.parameters["c"][1]))
        values[1] = fsp.omega_n * (np.cos(fsp.theta_n) * (x[0] - rmsh.parameters["c"][0]) - np.sin(fsp.theta_n) * (x[1] - rmsh.parameters["c"][1]))
  
    def value_shape(self):
        return (2,)

fsp.v__profile_l.interpolate(v__profile_l_Expression(element=fsp.Q_v_.ufl_element()))
fsp.v__profile_ellipse.interpolate(v__profile_ellipse_Expression(element=fsp.Q_v_.ufl_element()))


# BCs for `v_`, equivalent to (63) and (64) in 'Channel flow with moving obstacle'
bc_v__l = DirichletBC(fsp.Q_v_, fsp.v__profile_l, rmsh.mf, rmsh.parameters['line_l_id'])
bc_v__t = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.mf, rmsh.parameters['line_t_id'])
bc_v__b = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.mf, rmsh.parameters['line_b_id'])
bc_v__ellipse = DirichletBC(fsp.Q_v_, fsp.v__profile_ellipse, rmsh.mf, rmsh.parameters['polygon_id'])

bc_v_ = [bc_v__l, bc_v__t, bc_v__b, bc_v__ellipse]


# BCs for phi, equivalent fo (69) in 'Channel flow with moving obstacle

bc_phi_r = DirichletBC(fsp.Q_phi, Constant(0), rmsh.mf, rmsh.parameters['line_r_id'])
bc_phi = [bc_phi_r]

# BCs for v_n
bc_v_n = []


# Define variational problem for step 1
# step 1 for v_
F_v_ = ( \
                   rpam.parameters["rho"] * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_dot_n_1[k]) * ela.G(fsp.u_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_dot_n_2[k]) * ela.G(fsp.u_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + ela.G(fsp.u_n_1)[k, j] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n_1, rpam.parameters['mu'])[i, j] * (fsp.nu_v_[i]).dx(k) \
           ) * ela.detF(fsp.u_n_1) * rmsh.dx \
       - ( \
            ela.G(fsp.u_n_1)[k, j] * bgeo.facet_normal[k] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n_1, rpam.parameters['mu'])[i, j] * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_l_tb_poly \
            + ela.G(fsp.u_n_1)[k, 1] * bgeo.facet_normal[k] * flu.sigma_ale(fsp.V, fsp.sigma_n_32, fsp.u_n_1, rpam.parameters['mu'])[i, 1] * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_r \
        )

# sign

# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.parameters["rho"] / dt) * ela.G(fsp.u_n_1)[j, i] * ((fsp.v_[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx \
        + (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_n_1) * rmsh.ds_r

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.parameters["rho"]) * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l))) * fsp.nu_v_n[i]) * ela.detF(fsp.u_n_1) * rmsh.dx
