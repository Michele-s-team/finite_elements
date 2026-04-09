'''
this module solves for the field c^n  which define the state of M
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import physics.fluid_mechanics as flu
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


class D_c_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['D']

    def value_shape(self):
        return (1,)

fsp.D_c.interpolate(D_c_Expression(element=fsp.Q_c.ufl_element()))


bc_M = []

# variational problem

F_c = ( \
        ( (fsp.c_n - fsp.c_n_1)/dt  - ela.G(fsp.u_n_1_sq)[alpha, beta] * fsp.u_n_1_sq_dot[beta] * fsp.c_n.dx(alpha) ) * fsp.nu_c + \
        ela.G(fsp.u_n_1_sq)[gamma, alpha] * (fsp.D_c * ela.G(fsp.u_n_1_sq)[beta, alpha] * fsp.c_n.dx(beta) - fsp.v_square_n[alpha] * fsp.c_n) * fsp.nu_c.dx(gamma)              
    ) * ela.detF(fsp.u_n_1_sq) * rmsh.dx_sub_mesh[0][1] \
    + ( ela.G(fsp.u_n_1_sq)[gamma, alpha] * bgeo.sub_mesh_facet_normal[0][1][gamma] * fsp.v_square_n[alpha] * fsp.c_n * fsp.nu_c ) * ela.detF(fsp.u_n_1_sq) * rmsh.ds_sub_mesh[0][1]['ds_lrtb'] \
    + ( - rpam.parameters['k'] / ela.detF(fsp.u_n_1_sq) +  ela.G(fsp.u_n_1_sq)[gamma, alpha] * bgeo.sub_mesh_facet_normal[0][1][gamma] * fsp.v_square_n[alpha] * fsp.c_n ) * fsp.nu_c * ela.detF(fsp.u_n_1_sq) * rmsh.ds_sub_mesh[0][1]['ds_shape'] 

