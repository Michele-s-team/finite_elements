'''
this variational problem corresponds to the ODE
u - u_0 = dt * (\vec{v} \cdot \hat{n}(u)) \hat{n}(u)
with periodic BCs u(x_l) = u(x_r)
'''


from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import parameters.read.solution as rpam


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

alpha, beta = ufl.indices(2)

dt = rpam.parameters['T'] / rpam.parameters['N']

# the velocity profile v_expression must be a periodic function of x[0] to be consistent with the periodicity of the problem 
class v_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1
        values[1] = np.sin(2.0 * np.pi * (x[0] - rmsh.parameters['x_l']) / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)
    

class u_n_1_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    

class ys_expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0 * np.pi * (x[0] - rmsh.parameters['x_l']) / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        values[1] = np.sin(2.0 * np.pi * (x[0] - rmsh.parameters['x_l']) / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)

# fsp.u_n_1.interpolate(u_n_1_expression(element=fsp.Q.ufl_element()))

fsp.v.interpolate(v_expression(element=fsp.Q.ufl_element()))
fsp.ys.interpolate(ys_expression(element=fsp.Q.ufl_element()))

# fsp.u_n.assign(fsp.u_n_1)


# no BCs are needed here: the periodic BC is already implemented through the periodicity of the function space
bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F = (fsp.u_n[alpha] - fsp.u_n_1[alpha] - dt * (fsp.v[beta] * bgeo.n_ale(fsp.ys, fsp.u_n)[beta]) * bgeo.n_ale(fsp.ys, fsp.u_n)[alpha]) * \
    (
        fsp.nu_u_n[alpha] - \
        dt * (
            fsp.v[beta] * bgeo.delta_n_ale(fsp.ys, fsp.u_n, fsp.nu_u_n)[beta] * bgeo.n_ale(fsp.ys, fsp.u_n)[alpha] + \
            fsp.v[beta] * bgeo.n_ale(fsp.ys, fsp.u_n)[beta] * bgeo.delta_n_ale(fsp.ys, fsp.u_n, fsp.nu_u_n)[alpha]
        )
    ) * rmsh.dx
 