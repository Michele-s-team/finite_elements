'''
this module solves for the fields, v_disk^n, sigma_disk,...  which define the state of the disk fluid
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

alpha, beta, gamma, delta, epsilon, mu, nu, rho, sigma, tau = ufl.indices(10)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


class f_di_n_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 1

    def value_shape(self):
        return (2,)

fsp.f_di_n.interpolate(f_di_n_expression(element=fsp.Q_v_disk.ufl_element()))

'''
f_M = {\textrm{f}_M}_notes

Input values: 
    - 'c': concentration of M 'pulled back' to reference coordinates
    - 'U': displacement field of I (2-dimensional vector)

Return values: 
    - f_M (2-dimensional vector)
'''
def f_M(c, U):
    return as_tensor(0 * U[alpha], (alpha))


bc_v_disk__ = []
bc_phi_omega_disk = []
bc_v_disk_n = []



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

