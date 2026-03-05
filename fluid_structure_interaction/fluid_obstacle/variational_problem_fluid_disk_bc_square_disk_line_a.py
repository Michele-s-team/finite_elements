'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
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

alpha, beta, gamma, delta = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size


bc_v_disk__ = []
bc_phi_disk = []



# Define variational problem for step 1
# step 1 for v_
# natural BC imposed here
F_v_disk__ = ( \
                   rpam.parameters['rho_fluid'] * ((fsp.v_disk__[alpha] - fsp.v_disk_n_1[alpha]) / dt \
                               + (3.0 / 2.0 * (fsp.v_disk_n_1[gamma] - fsp.u_n_1_di_dot[gamma]) * ela.G(fsp.u_n_1_di)[beta, gamma] - 1.0 / 2.0 * (fsp.v_disk_n_2[gamma] - fsp.u_n_2_di_dot[gamma]) * ela.G(fsp.u_n_2_di)[beta, gamma]) * (fsp.V_di[alpha]).dx(beta)
#    sign
                               ) * fsp.nu_v_[alpha] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_msh_n_1)[delta, alpha] * (fsp.nu_v_[alpha]).dx(delta) + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[gamma, beta] * ((fsp.V[alpha]).dx(gamma)) * ela.G(fsp.u_msh_n_1)[delta, beta] * (fsp.nu_v_[alpha]).dx(delta) \
           ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1] \
       - (ela.G(fsp.u_msh_n_1)[delta, alpha] * bgeo.sub_mesh_facet_normal[1][delta] * fsp.sigma_n_32 * fsp.nu_v_[alpha]) * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds']  \
       - ( \
                   rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[delta, beta] * bgeo.sub_mesh_facet_normal[1][delta] * ela.G(fsp.u_msh_n_1)[gamma, beta] * (fsp.V[alpha].dx(gamma)) * fsp.nu_v_[alpha] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_l']\
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[delta, beta] * bgeo.sub_mesh_facet_normal[1][delta] * ela.G(fsp.u_msh_n_1)[gamma, beta] * (fsp.V[alpha].dx(gamma)) * fsp.nu_v_[alpha] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_tb'] \
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[delta, beta] * bgeo.sub_mesh_facet_normal[1][delta] * ela.G(fsp.u_msh_n_1)[gamma, beta] * (fsp.V[alpha].dx(gamma)) * fsp.nu_v_[alpha] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_ellipse'] \
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[delta, 1] * bgeo.sub_mesh_facet_normal[1][delta] * ela.G(fsp.u_msh_n_1)[gamma, 1] * (fsp.V[alpha].dx(gamma)) * fsp.nu_v_[alpha] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_r'] \
           )


'''
# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_msh_n_1)[beta, alpha] * (fsp.phi.dx(beta)) * ela.G(fsp.u_msh_n_1)[delta, alpha] * (fsp.nu_phi.dx(delta)) \
                    - (rpam.parameters['rho_fluid'] / dt) * ela.G(fsp.u_msh_n_1)[beta, alpha] * ((fsp.v_[alpha]).dx(beta)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1] \
        + (ela.G(fsp.u_msh_n_1)[delta, alpha] * bgeo.sub_mesh_facet_normal[1][delta] * ela.G(fsp.u_msh_n_1)[beta, alpha] * (fsp.phi.dx(beta)) * fsp.nu_phi) * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_r']


# step 3 for v_n
F_v_n = ( ( (fsp.v_n[alpha] - fsp.v_[alpha]) + (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_msh_n_1)[delta, alpha] * (fsp.phi.dx(delta)) ) * fsp.nu_v_n[alpha] ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1]
'''