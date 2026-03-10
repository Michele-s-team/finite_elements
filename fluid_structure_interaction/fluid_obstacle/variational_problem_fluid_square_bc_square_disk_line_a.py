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


class f_sq_n_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = - rpam.parameters['rho_sq'] * rpam.parameters['g']

    def value_shape(self):
        return (2,)

fsp.f_sq_n.interpolate(f_sq_n_expression(element=fsp.Q_v_square.ufl_element()))


class t_sq_n_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

fsp.t_sq_n.interpolate(t_sq_n_expression(element=fsp.Q_v_square.ufl_element()))


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
    DirichletBC(fsp.Q_v__square, fsp.v_square__bc, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q_v__square, fsp.v_disk_n_0_0_on_0_1, rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["circle_id"])\
    ]

bc_phi_square = [
        DirichletBC(fsp.Q_sigma_square, Constant(0), rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"])\
]

bc_v_square_n = []



# Define variational problem for step 1
# step 1 for v_
# natural BC imposed here, and I dropped the boundary terms where Dirichlet BCs are imposed because the test function vanishes on those boundaries
F_v_square__ = (
                ( rpam.parameters['rho_sq'] * ( 
                        (fsp.v_square__[alpha] - fsp.v_square_n_1[alpha]) / dt \
                        + ( 3.0 / 2.0 * (fsp.v_square_n_1[gamma] - fsp.u_n_1_sq_dot[gamma] ) * ela.G(fsp.u_n_1_sq)[beta, gamma] - 1.0 / 2.0 * ( fsp.v_square_n_2[gamma] - fsp.u_n_2_sq_dot[gamma] ) * ela.G(fsp.u_n_2_sq)[beta, gamma] ) * (fsp.V_sq[alpha]).dx(beta) 
                    )
                    - fsp.f_sq_n[alpha]
                ) * fsp.nu_v_square__[alpha] + \
                ela.G(fsp.u_n_1_sq)[gamma, beta] *  flu.sigma_ale(fsp.V_sq, fsp.sigma_square_n_32, fsp.u_n_1_sq, rpam.parameters['eta_sq'])[alpha, beta] * fsp.nu_v_square__[alpha].dx(gamma) \
            ) * ela.detF(fsp.u_n_1_sq) * rmsh.dx_sub_mesh[0][1] \
       - ( 1.0/ela.detF(fsp.u_n_1_sq) * fsp.t_sq_n[alpha] ) * fsp.nu_v_square__[alpha] *  ela.detF(fsp.u_n_1_sq) * rmsh.ds_sub_mesh[0][1]['ds_t'] 




# step 2 for phi_square
F_phi_square = ( \
                    - ela.G(fsp.u_n_1_sq)[beta, alpha] * (fsp.phi_square.dx(beta)) * ela.G(fsp.u_n_1_sq)[delta, alpha] * (fsp.nu_phi_square.dx(delta)) \
                    - (rpam.parameters['rho_sq'] / dt) * ela.G(fsp.u_n_1_sq)[beta, alpha] * ((fsp.v_square__[alpha]).dx(beta)) * fsp.nu_phi_square \
            ) * ela.detF(fsp.u_n_1_sq) * rmsh.dx_sub_mesh[0][1] \
        + (ela.G(fsp.u_n_1_sq)[delta, alpha] * bgeo.sub_mesh_facet_normal[0][1][delta] * ela.G(fsp.u_n_1_sq)[beta, alpha] * (fsp.phi_square.dx(beta)) * fsp.nu_phi_square) * ela.detF(fsp.u_n_1_sq) * rmsh.ds_sub_mesh[0][1]['ds_t'] 




# step 3 for v_square_n
F_v_square_n = ( ( (fsp.v_square_n[alpha] - fsp.v_square__[alpha]) + (dt / rpam.parameters['rho_sq']) * ela.G(fsp.u_n_1_sq)[gamma, alpha] * (fsp.phi_square.dx(gamma)) ) * fsp.nu_v_square_n[alpha] ) * ela.detF(fsp.u_n_1_sq) * rmsh.dx_sub_mesh[0][1]


# check
