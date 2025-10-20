'''
this module solves for the variables u, \dot{u} which define the state of the mesh
'''

from fenics import *
import importlib
import ufl as ufl

import elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta = ufl.indices(2)

# BCs
# BCs for u
bc_u_l = DirichletBC(fsp.Q_u, Constant((0, 0)), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_l_id"])
bc_u_b = DirichletBC(fsp.Q_u, Constant((0, 0)), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_b_id"])

bc_u_0_r = DirichletBC(fsp.Q_u.sub(0), Constant(0), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_r_id"])

bc_u_t = DirichletBC(fsp.Q_u, fsp.U_n_12_on_mesh, rmsh.mf_sub_mesh[0], rmsh.parameters["sub_mesh_1_id"])


bcs_u = [bc_u_l, bc_u_b, bc_u_0_r, bc_u_t]


# BCs for u_dot
bc_u_dot_l = DirichletBC(fsp.Q_u_dot, Constant((0, 0)), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_l_id"])
bc_u_dot_b = DirichletBC(fsp.Q_u_dot, Constant((0, 0)), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_b_id"])

bc_u_dot_0_r = DirichletBC(fsp.Q_u_dot.sub(0), Constant(0), rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_r_id"])

bc_u_dot_t = DirichletBC(fsp.Q_u_dot, fsp.U_dot_n_12_on_mesh, rmsh.mf_sub_mesh[0], rmsh.parameters["sub_mesh_1_id"])


bcs_u_dot = [bc_u_dot_l, bc_u_dot_b, bc_u_dot_0_r, bc_u_dot_t]

# sign




# variational functional for the original problem
F_msh_u = (ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[alpha, beta] * (fsp.nu_u[alpha].dx(beta))) * rmsh.dx_sub_mesh[1]

F_msh_u_dot = ( \
                          (ela.F_dot(fsp.u_dot_n)[alpha, j] * ela.S(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[j, beta] \
                           + ela.F(fsp.u_n)[alpha, j] * ela.S_dot(fsp.u_n,
                                                                  fsp.u_dot_n,
                                                                  ela.K(fsp.u_n, rpam.parameters['exponent']),
                                                                  ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']),
                                                                  ela.mu(fsp.u_n, rpam.parameters['exponent']),
                                                                  ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']))[j, beta]) \
                          * (fsp.nu_u_dot[alpha].dx(beta))) * rmsh.dx_sub_mesh[1]


