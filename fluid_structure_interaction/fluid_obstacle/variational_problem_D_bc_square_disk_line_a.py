'''
this module solves for the fields u_n_di, u_n_di_dot, u_n_sq, u_n_sq_dot which set the D (domain displacement)
'''

from fenics import *
import importlib
import ufl as ufl

import elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)


bcs_u_sq = [ \
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q_u_sq, fsp.U_n_12_1_on_0_1, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]

# sign


bcs_u_sq_dot = [ \
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    # sign
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]

'''
bc_u_sq_square = DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.boundary[1]['lrtb'])

bc_u_msh_ellipse = DirichletBC(fsp.Q_u_msh, fsp.u_el_n_on_sub_mesh_1, rmsh.boundary[1]['ellipse'])
bcs_msh = [bc_u_msh_ellipse, bc_u_msh_square]

# variational functional for the original problem
F_msh_u = (ela.P(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.parameters['exponent']), ela.mu(fsp.u_msh_n, rpam.parameters['exponent']))[gamma, alpha] * (fsp.nu_u_msh[gamma].dx(alpha))) * rmsh.dx_sub_mesh[1]

bc_u_msh_dot_ellipse = DirichletBC(fsp.Q_u_msh_dot, fsp.u_el_dot_n_on_sub_mesh_1, rmsh.boundary[1]['ellipse'])
bc_u_msh_dot_square = DirichletBC(fsp.Q_u_msh_dot, Constant((0, 0)), rmsh.boundary[1]['lrtb'])
bcs_msh_dot = [bc_u_msh_dot_ellipse, bc_u_msh_dot_square]

F_msh_u_dot = ( \
                          (ela.F_dot(fsp.u_msh_dot_n)[gamma, beta] * ela.S(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.parameters['exponent']), ela.mu(fsp.u_msh_n, rpam.parameters['exponent']))[beta, alpha] \
                           + ela.F(fsp.u_msh_n)[gamma, beta] * ela.S_dot(fsp.u_msh_n,
                                                                  fsp.u_msh_dot_n,
                                                                  ela.K(fsp.u_msh_n, rpam.parameters['exponent']),
                                                                  ela.K_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.parameters['exponent']),
                                                                  ela.mu(fsp.u_msh_n, rpam.parameters['exponent']),
                                                                  ela.mu_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.parameters['exponent']))[beta, alpha]) \
                          * (fsp.nu_u_msh_dot[gamma].dx(alpha))) * rmsh.dx_sub_mesh[1]
'''