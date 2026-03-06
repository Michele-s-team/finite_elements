'''
this module solves for the fields, v_square^n, sigma_square,...  which define the state of the square fluid
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import fluid as flu
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


class f_sq_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

fsp.f_sq_n.interpolate(f_sq_expression(element=fsp.Q_v_square.ufl_element()))

# sign

'''
v_square__bc_Expression =  g_notes used to enforce the BCs for v_square__ on \partial \Omega^y_{sq IN} U \partial \Omega^y_{sq OUT} U \partial \Omega^y_{sq B}
'''
class v_square__bc_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

fsp.v_square__bc.interpolate(v_square__bc_Expression(element=fsp.Q_v__square.ufl_element()))


bc_v_square__ = [
    DirichletBC(fsp.Q_v__square, fsp.v_square__bc, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q_v__square, fsp.v_square__bc, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q_v__square, fsp.v_square__bc, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q_v__square, fsp.v_square__bc, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q_v__square, fsp.v_disk_n_0_0_on_0_1, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])\
    ]

'''
bc_phi_square = []
bc_v_square_n = []



# Define variational problem for step 1
# step 1 for v_
# natural BC imposed here
F_v_disk__ = (
              ( rpam.parameters['rho_di'] * ( (fsp.v_disk__[alpha] - fsp.v_disk_n_1[alpha]) / dt \
                               + (3.0 / 2.0 * (fsp.v_disk_n_1[gamma] - fsp.u_n_1_di_dot[gamma]) * ela.G(fsp.u_n_1_di)[beta, gamma] - 1.0 / 2.0 * (fsp.v_disk_n_2[gamma] - fsp.u_n_2_di_dot[gamma]) * ela.G(fsp.u_n_2_di)[beta, gamma] ) * (fsp.V_di[alpha]).dx(beta) )
                               - fsp.f_di_n[alpha]
                               ) * fsp.nu_v_disk__[alpha] \
                   + ela.G(fsp.u_n_1_di)[gamma, beta] *  flu.sigma_ale(fsp.V_di, fsp.sigma_disk_n_32, fsp.u_n_1_di, rpam.parameters['eta_di'])[alpha, beta] * fsp.nu_v_disk__[alpha].dx(gamma) \
            ) * ela.detF(fsp.u_n_1_di) * rmsh.dx_sub_mesh[0][0] \
       - ( flu.sigma_ale(fsp.v_square_n_1_0_1_on_0_0, fsp.sigma_square_n_32_0_1_on_0_0, fsp.u_n_1_di, rpam.parameters['eta_sq'])[alpha, beta] * ela.G(fsp.u_n_1_di)[gamma, beta] * bgeo.sub_mesh_facet_normal[0][0][gamma] + 1.0 / ela.detF(fsp.u_n_1_di) * f_M(fsp.c_n_1, fsp.U_n_32)[alpha] ) * fsp.nu_v_disk__[alpha] *  ela.detF(fsp.u_n_1_di) * rmsh.ds_sub_mesh[0][0]['ds'] 



# step 2 for phi_disk
F_phi_disk = ( \
                    - ela.G(fsp.u_n_1_di)[beta, alpha] * (fsp.phi_disk.dx(beta)) * ela.G(fsp.u_n_1_di)[delta, alpha] * (fsp.nu_phi_disk.dx(delta)) \
                    - (rpam.parameters['rho_di'] / dt) * ela.G(fsp.u_n_1_di)[beta, alpha] * ((fsp.v_disk__[alpha]).dx(beta)) * fsp.nu_phi_disk \
            ) * ela.detF(fsp.u_n_1_di) * rmsh.dx_sub_mesh[0][0] \
        + (ela.G(fsp.u_n_1_di)[delta, alpha] * bgeo.sub_mesh_facet_normal[0][0][delta] * ela.G(fsp.u_n_1_di)[beta, alpha] * (fsp.phi_disk.dx(beta)) * fsp.nu_phi_disk) * ela.detF(fsp.u_n_1_di) * rmsh.ds_sub_mesh[0][0]['ds'] 



F_omega_disk = ( fsp.omega_disk[alpha] - ela.G(fsp.u_n_1_di)[beta, alpha] * fsp.phi_disk.dx(beta) ) * ( fsp.nu_omega_disk[alpha] - ela.G(fsp.u_n_1_di)[gamma, alpha] * fsp.nu_phi_disk.dx(gamma) ) * ela.detF(fsp.u_n_1_di) * rmsh.dx_sub_mesh[0][0]


F_N =  rpam.parameters['alpha'] / rmsh.r_sub_mesh[0][0] * (
        - fsp.phi_disk * ela.G(fsp.u_n_1_di)[beta, alpha] * bgeo.sub_mesh_facet_normal[0][0][beta] * ela.G(fsp.u_n_1_di)[gamma, alpha] * bgeo.sub_mesh_facet_normal[0][0][gamma] + 
        ela.G(fsp.u_n_1_di)[delta, alpha] * bgeo.sub_mesh_facet_normal[0][0][delta] * 
            (
                flu.sigma_ale(fsp.V_di, fsp.sigma_disk_n_32, fsp.u_n_1_di, rpam.parameters['eta_di'])[alpha, beta] * ela.G(fsp.u_n_1_di)[gamma, beta] * bgeo.sub_mesh_facet_normal[0][0][gamma] -
                (
                    flu.sigma_ale(fsp.v_square_n_1_0_1_on_0_0, fsp.sigma_square_n_32_0_1_on_0_0, fsp.u_n_1_di, rpam.parameters['eta_sq'])[alpha, beta] *  ela.G(fsp.u_n_1_di)[gamma, beta] *  bgeo.sub_mesh_facet_normal[0][0][gamma] +  
                    1.0/ela.detF(fsp.u_n_1_di) * f_M(fsp.c_n_1, fsp.U_n_32)[alpha] 
                ) - 
                    rpam.parameters['eta_di'] * dt / rpam.parameters['rho_di'] * ela.G(fsp.u_n_1_di)[gamma, beta] *  bgeo.sub_mesh_facet_normal[0][0][gamma] * ela.G(fsp.u_n_1_di)[epsilon, beta] * fsp.omega_disk[alpha].dx(epsilon)
            )
        ) * \
        (
            - fsp.nu_phi_disk * ela.G(fsp.u_n_1_di)[nu, mu] *  bgeo.sub_mesh_facet_normal[0][0][nu] *  ela.G(fsp.u_n_1_di)[rho, mu] *  bgeo.sub_mesh_facet_normal[0][0][rho] - \
            rpam.parameters['eta_di'] * dt / rpam.parameters['rho_di'] * ela.G(fsp.u_n_1_di)[sigma, mu] *  bgeo.sub_mesh_facet_normal[0][0][sigma] * ela.G(fsp.u_n_1_di)[rho, nu] *  bgeo.sub_mesh_facet_normal[0][0][rho] * ela.G(fsp.u_n_1_di)[tau, nu] * ((fsp.nu_omega_disk)[mu]).dx(tau)
        ) * \
    ela.detF(fsp.u_n_1_di) * ela.detF(fsp.u_n_1_di) * rmsh.ds_sub_mesh[0][0]['ds'] 


F_phi_omega_disk = (F_phi_disk + F_omega_disk) + F_N



# step 3 for v_n
F_v_disk_n = ( ( (fsp.v_disk_n[alpha] - fsp.v_disk__[alpha]) + (dt / rpam.parameters['rho_di']) * ela.G(fsp.u_n_1_di)[gamma, alpha] * (fsp.phi_disk.dx(gamma)) ) * fsp.nu_v_disk_n[alpha] ) * ela.detF(fsp.u_n_1_di) * rmsh.dx_sub_mesh[0][0]

'''