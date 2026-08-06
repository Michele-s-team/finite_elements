'''
this module solves for the variables u, \dot{u} which define the state of the mesh
'''

from fenics import *
import importlib
import ufl as ufl
from ufl import dot
from fenics import project
import numpy as np

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma = ufl.indices(3)


# BCs
# BCs for u
# bc_u_l = DirichletBC(fsp.Q_u, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
# bc_u_0_l = DirichletBC(fsp.Q_u.sub(0),Constant(0),rmsh.lmsh.mf_sub_meshes[0],rmsh.parameters["line_sub_mesh_0_l_id"])


bc_u_0_l = DirichletBC(fsp.Q_u.sub(0), Constant(0), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
bc_u_0_r = DirichletBC(fsp.Q_u.sub(0), Constant(0), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_r_id"])

bc_u_t = DirichletBC(fsp.Q_u, Constant((0,0)), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"])

# bc_u_b = DirichletBC(fsp.Q_u, fsp.u_fs_on_mesh, rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_2_id"])



'''boundary condition for the bottom needs to be removed in order to let it deform'''
# bc_u_b = DirichletBC(fsp.Q_u, fsp.u_fs_on_mesh, rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_2_id"])


bcs_msh = [ bc_u_0_l, bc_u_0_r, bc_u_t]




# BCs for u_dot
# bc_u_dot_b = DirichletBC(fsp.Q_u_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
#bc_u_dot_l = Dirich.letBC(fsp.Q_u_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_b_id"])

'''free surface velocity equal to normal velocity. mesh_id changed into the one for the free boundary'''
bc_u_dot_b = DirichletBC(fsp.Q_u_dot, Constant(1)*fsp.v_fl_n,  rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_2_id"])

''' y component of the left boundary is set free(removing the fixed membrane boundary on the left), in order to apply the periodic boundary conditions for left and right) '''
bc_u_dot_0_l = DirichletBC(fsp.Q_u_dot.sub(0), Constant(0),rmsh.lmsh.mf_sub_meshes[0],rmsh.parameters["line_sub_mesh_0_l_id"])
bc_u_dot_0_r = DirichletBC(fsp.Q_u_dot.sub(0), Constant(0), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_r_id"])

''' top boundary is left unmoving, both x and y velocity components are set to 0 '''
bc_u_dot_t = DirichletBC(fsp.Q_u_dot, Constant((0, 0)), rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"])

# bc_u_dot_t = DirichletBC(fsp.Q_u_dot,  fsp.U_dot_n_12_on_mesh,rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"])
bcs_msh_dot = [bc_u_dot_0_l, bc_u_dot_b, bc_u_dot_0_r, bc_u_dot_t]





# variational functional for u
F_u = - (ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[alpha, beta] \
         * (fsp.nu_u[alpha].dx(beta))) * rmsh.dx_sub_mesh[0] \
    + ((bgeo.sub_mesh_facet_normal[0])[beta] * ela.P(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[alpha, beta] \
       * fsp.nu_u[alpha]) * rmsh.ds_sub_mesh[0]['ds']


#du_n/d_x[0] = 0
F_u_N = rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( (fsp.u_n[1].dx(0)) * fsp.nu_u[1].dx(0) ) * rmsh.ds_sub_mesh[0]['ds_lr']    
        
F_msh = F_u + F_u_N


# variational problem for u_dot
F_u_dot = - ( \
                    (
                        ela.F_dot(fsp.u_dot_n)[alpha, gamma] * ela.S(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[gamma, beta] \
                        + ela.F(fsp.u_n)[alpha, gamma] * ela.S_dot(
                                                                    fsp.u_n,
                                                                    fsp.u_dot_n,
                                                                    ela.K(fsp.u_n, rpam.parameters['exponent']),
                                                                    ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']),
                                                                    ela.mu(fsp.u_n, rpam.parameters['exponent']),
                                                                    ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent'])
                                                                )[gamma, beta]
                    ) * (fsp.nu_u_dot[alpha].dx(beta))\
                ) * rmsh.dx_sub_mesh[0] \
                + (\
                    (bgeo.sub_mesh_facet_normal[0])[beta] * (
                        ela.F_dot(fsp.u_dot_n)[alpha, gamma] * ela.S(fsp.u_n, ela.K(fsp.u_n, rpam.parameters['exponent']), ela.mu(fsp.u_n, rpam.parameters['exponent']))[gamma, beta] \
                        + ela.F(fsp.u_n)[alpha, gamma] * ela.S_dot(
                                                                    fsp.u_n,
                                                                    fsp.u_dot_n,
                                                                    ela.K(fsp.u_n, rpam.parameters['exponent']),
                                                                    ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent']),
                                                                    ela.mu(fsp.u_n, rpam.parameters['exponent']),
                                                                    ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters['exponent'])
                                                                )[gamma, beta]
                    )    
                ) * fsp.nu_u_dot[alpha] * rmsh.ds_sub_mesh[0]['ds']


# F_u_dot_N = rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( (fsp.u_dot_n[1].dx(0)) * fsp.nu_u_dot[1].dx(0) ) * rmsh.ds_sub_mesh[0]['ds_r']  

F_u_dot_N = rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( (fsp.u_dot_n[1].dx(0)) * fsp.nu_u_dot[1].dx(0) ) * rmsh.ds_sub_mesh[0]['ds_lr']  
'''changed ds_r into ds'''  

F_msh_dot = F_u_dot + F_u_dot_N

