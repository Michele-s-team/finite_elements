'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import numpy as np
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.parameters["T"] / rpam.parameters["num_steps"]  # time step size

focus = cal.ellipse_focal_points(rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'])[0]





v__profile_l = Expression((f'{rpam.parameters["v_l"]}* 4.0*1.5*x[1]*({rmsh.parameters["h"]} - x[1]) / pow({rmsh.parameters["h"]}, 2)', '0'), element=fsp.Q_v_.ufl_element(), h=rmsh.parameters["h"])
bc_v__l = DirichletBC(fsp.Q_v_, v__profile_l, rmsh.mf, rmsh.parameters['line_l_id'])

bc_v__t = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.mf, rmsh.parameters['line_t_id'])
bc_v__b = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.mf, rmsh.parameters['line_b_id'])

v__profile_ellipse = Expression((f'{fsp.omega_n} * (-sin({fsp.theta_n}) * (x[0] - {focus[0]}) - cos({fsp.theta_n}) * (x[1] - {focus[1]}))', f'{fsp.omega_n} * (cos({fsp.theta_n}) * (x[0] - {focus[0]}) - sin({fsp.theta_n}) * (x[1] - {focus[1]}))'), element=fsp.Q_v_.ufl_element())
bc_v__ellipse = DirichletBC(fsp.Q_v_, v__profile_ellipse, rmsh.mf, rmsh.parameters['polygon_id'])

bc_phi_r = DirichletBC(fsp.Q_phi, Constant(0), rmsh.mf, rmsh.parameters['line_r_id'])

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__l, bc_v__t, bc_v__b, bc_v__ellipse]
bc_phi = [bc_phi_r]
bc_v_n = []

# Define variational problem for step 1
# step 1 for v_
F_v_ = ( \
                   rpam.parameters["rho"] * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_dot_n_1[k]) * ela.G(fsp.u_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_dot_n_2[k]) * ela.G(fsp.u_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_v_[i]).dx(l) + rpam.parameters["mu"] * ela.G(fsp.u_n_1)[k, j] * ((fsp.V[i]).dx(k)) * ela.G(fsp.u_n_1)[l, j] * (fsp.nu_v_[i]).dx(l) \
           ) * ela.detF(fsp.u_n_1) * rmsh.dx \
       - (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * fsp.sigma_n_32 * fsp.nu_v_[i]) * ela.detF(fsp.u_n_1) * rmsh.ds \
       - ( \
                   rpam.parameters["mu"] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_l \
                   + rpam.parameters["mu"] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_tb \
                   + rpam.parameters["mu"] * ela.G(fsp.u_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_poly \
                   + rpam.parameters["mu"] * ela.G(fsp.u_n_1)[l, 1] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[k, 1] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_n_1) * rmsh.ds_r \
           )

# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.parameters["rho"] / dt) * ela.G(fsp.u_n_1)[j, i] * ((fsp.v_[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx \
        + (ela.G(fsp.u_n_1)[l, i] * bgeo.facet_normal[l] * ela.G(fsp.u_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_n_1) * rmsh.ds_r

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.parameters["rho"]) * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l))) * fsp.nu_v_n[i]) * ela.detF(fsp.u_n_1) * rmsh.dx
