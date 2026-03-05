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

# 1 BCs for square

bcs_u_sq = [ \
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q_u_sq, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q_u_sq, fsp.U_n_12_1_on_0_1, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]

bcs_u_sq_dot = [ \
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q_u_sq_dot, fsp.u_n_sq_dot_bc_di, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]


# 2 BCs for disk

bcs_u_di = [  
       DirichletBC(fsp.Q_u_di, fsp.U_n_12_1_on_0_0, rmsh.lmsh.mf_sub_meshes[0][0], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]

bcs_u_di_dot = [ \
        DirichletBC(fsp.Q_u_di_dot, fsp.u_n_di_dot_bc_di, rmsh.lmsh.mf_sub_meshes[0][0], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]


F_u_sq = (ela.P(fsp.u_n_sq, ela.K(fsp.u_n_sq, rpam.parameters['exponent']), ela.mu(fsp.u_n_sq, rpam.parameters['exponent']))[gamma, alpha] * (fsp.nu_u_n_sq[gamma].dx(alpha))) * rmsh.dx_sub_mesh[0][1]

F_u_di = (ela.P(fsp.u_n_di, ela.K(fsp.u_n_di, rpam.parameters['exponent']), ela.mu(fsp.u_n_di, rpam.parameters['exponent']))[gamma, alpha] * (fsp.nu_u_n_di[gamma].dx(alpha))) * rmsh.dx_sub_mesh[0][0]


F_u_sq_dot = ( \
                          (ela.F_dot(fsp.u_n_sq_dot)[gamma, beta] * ela.S(fsp.u_n_sq, ela.K(fsp.u_n_sq, rpam.parameters['exponent']), ela.mu(fsp.u_n_sq, rpam.parameters['exponent']))[beta, alpha] \
                           + ela.F(fsp.u_n_sq)[gamma, beta] * ela.S_dot(fsp.u_n_sq,
                                                                  fsp.u_n_sq_dot,
                                                                  ela.K(fsp.u_n_sq, rpam.parameters['exponent']),
                                                                  ela.K_dot(fsp.u_n_sq, fsp.u_n_sq_dot, rpam.parameters['exponent']),
                                                                  ela.mu(fsp.u_n_sq, rpam.parameters['exponent']),
                                                                  ela.mu_dot(fsp.u_n_sq, fsp.u_n_sq_dot, rpam.parameters['exponent']))[beta, alpha]) \
                          * (fsp.nu_u_n_sq_dot[gamma].dx(alpha))) * rmsh.dx_sub_mesh[0][1]

F_u_di_dot = ( \
                          (ela.F_dot(fsp.u_n_di_dot)[gamma, beta] * ela.S(fsp.u_n_di, ela.K(fsp.u_n_di, rpam.parameters['exponent']), ela.mu(fsp.u_n_di, rpam.parameters['exponent']))[beta, alpha] \
                           + ela.F(fsp.u_n_di)[gamma, beta] * ela.S_dot(fsp.u_n_di,
                                                                  fsp.u_n_di_dot,
                                                                  ela.K(fsp.u_n_di, rpam.parameters['exponent']),
                                                                  ela.K_dot(fsp.u_n_di, fsp.u_n_di_dot, rpam.parameters['exponent']),
                                                                  ela.mu(fsp.u_n_di, rpam.parameters['exponent']),
                                                                  ela.mu_dot(fsp.u_n_di, fsp.u_n_di_dot, rpam.parameters['exponent']))[beta, alpha]) \
                          * (fsp.nu_u_n_di_dot[gamma].dx(alpha))) * rmsh.dx_sub_mesh[0][0]

# sign
