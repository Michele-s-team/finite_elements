'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
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


class f_di_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 1

    def value_shape(self):
        return (2,)

fsp.f_di_n.interpolate(f_di_expression(element=fsp.Q_v_disk.ufl_element()))

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
bc_phi_disk = []



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

# sign


'''
# step 3 for v_n
F_v_n = ( ( (fsp.v_n[alpha] - fsp.v_[alpha]) + (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_msh_n_1)[delta, alpha] * (fsp.phi.dx(delta)) ) * fsp.nu_v_n[alpha] ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1]
'''