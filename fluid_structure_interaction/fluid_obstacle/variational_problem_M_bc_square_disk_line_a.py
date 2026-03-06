'''
this module solves for the field c^n  which define the state of M
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

bc_M = []

# variational problem
'''
F_phi_square = ( \
                    - ela.G(fsp.u_n_1_sq)[beta, alpha] * (fsp.phi_square.dx(beta)) * ela.G(fsp.u_n_1_sq)[delta, alpha] * (fsp.nu_phi_square.dx(delta)) \
                    - (rpam.parameters['rho_sq'] / dt) * ela.G(fsp.u_n_1_sq)[beta, alpha] * ((fsp.v_square__[alpha]).dx(beta)) * fsp.nu_phi_square \
            ) * ela.detF(fsp.u_n_1_sq) * rmsh.dx_sub_mesh[0][1] \
        + (ela.G(fsp.u_n_1_sq)[delta, alpha] * bgeo.sub_mesh_facet_normal[0][1][delta] * ela.G(fsp.u_n_1_sq)[beta, alpha] * (fsp.phi_square.dx(beta)) * fsp.nu_phi_square) * ela.detF(fsp.u_n_1_sq) * rmsh.ds_sub_mesh[0][1]['ds_t'] 
'''
