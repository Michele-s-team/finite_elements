'''
this module solves for the variables u, \dot{u} which define the state of the mesh
'''

from fenics import *
import importlib
import ufl as ufl

import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

bc_u_msh_ellipse = DirichletBC(fsp.Q_u_msh, fsp.u_el_n_on_sub_mesh_1, rmsh.boundary[1]['ellipse'])
bc_u_msh_square = DirichletBC(fsp.Q_u_msh, Constant((0, 0)), rmsh.boundary[1]['lrtb'])
bcs_msh = [bc_u_msh_ellipse, bc_u_msh_square]

# variational functional for the original problem
F_msh_u = (ela.P(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.parameters['exponent']), ela.mu(fsp.u_msh_n, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_msh[k].dx(i))) * rmsh.dx_sub_mesh[1]

bc_u_msh_dot_ellipse = DirichletBC(fsp.Q_u_msh_dot, fsp.u_el_dot_n_on_sub_mesh_1, rmsh.boundary[1]['ellipse'])
bc_u_msh_dot_square = DirichletBC(fsp.Q_u_msh_dot, Constant((0, 0)), rmsh.boundary[1]['lrtb'])
bcs_msh_dot = [bc_u_msh_dot_ellipse, bc_u_msh_dot_square]

F_msh_u_dot = ( \
                          (ela.F_dot(fsp.u_msh_dot_n)[k, j] * ela.S(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.parameters['exponent']), ela.mu(fsp.u_msh_n, rpam.parameters['exponent']))[j, i] \
                           + ela.F(fsp.u_msh_n)[k, j] * ela.S_dot(fsp.u_msh_n,
                                                                  fsp.u_msh_dot_n,
                                                                  ela.K(fsp.u_msh_n, rpam.parameters['exponent']),
                                                                  ela.K_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.parameters['exponent']),
                                                                  ela.mu(fsp.u_msh_n, rpam.parameters['exponent']),
                                                                  ela.mu_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.parameters['exponent']))[j, i]) \
                          * (fsp.nu_u_msh_dot[k].dx(i))) * rmsh.dx_sub_mesh[1]
