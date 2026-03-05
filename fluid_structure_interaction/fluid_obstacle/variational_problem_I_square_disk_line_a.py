'''
    this module solves the variational problem

    U^{n-1/2} - U^{n-3/2} = dt * (\vec{v} \cdot \hat{n}(U^{n-1/2})) \hat{n}(U^{n-1/2})

    with periodic BCs u(x_l) = u(x_r)
'''


from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)

alpha, beta = ufl.indices(2)

dt = rpam.parameters['T'] / rpam.parameters['N']

'''
# the velocity profile v_expression must be a periodic function of x[0] to be consistent with the periodicity of the problem 
class v_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1
        values[1] = np.sin(2.0 * np.pi * x[0] / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)
    

class u_n_1_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


fsp.v.interpolate(v_expression(element=fsp.Q.ufl_element()))
'''

class ys_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rmsh.lmsh.parameters['c_r'][0] + rmsh.lmsh.parameters['r'] * np.cos(2.0 * np.pi * (x[0] - rmsh.lmsh.mesh_parameters[1]['x_l']) / (rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l']))
        values[1] = rmsh.lmsh.parameters['c_r'][1] + rmsh.lmsh.parameters['r'] * np.sin(2.0 * np.pi * (x[0] - rmsh.lmsh.mesh_parameters[1]['x_l']) / (rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l']))

    def value_shape(self):
        return (2,)

fsp.ys.interpolate(ys_expression(element=fsp.Q_U.ufl_element()))


# no BCs are needed here: the periodic BC is already implemented through the periodicity of the function space
bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F_U = (fsp.U_n_12[alpha] - fsp.U_n_32[alpha] - dt * (fsp.v_square_n_1_0_1_on_1[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[beta]) * bgeo.n_ale(fsp.ys, fsp.U_n_12)[alpha]) * \
    (
        fsp.nu_U[alpha] - \
        dt * (
            fsp.v_square_n_1_0_1_on_1[beta] * bgeo.delta_n_ale(fsp.ys, fsp.U_n_12, fsp.nu_U)[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[alpha] + \
            fsp.v_square_n_1_0_1_on_1[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[beta] * bgeo.delta_n_ale(fsp.ys, fsp.U_n_12, fsp.nu_U)[alpha]
        )
    ) * rmsh.dx_mesh[1]
 

#  sign