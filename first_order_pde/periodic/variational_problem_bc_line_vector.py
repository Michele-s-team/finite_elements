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


class v_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['dt'] * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))**2
        values[1] = 0

    def value_shape(self):
        return (2,)
    

class u_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.1 * np.cos(4*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        values[1] = 0.1 * np.sin(8*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)
    

class ys_expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0 * np.pi * x[0] / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        values[1] = np.sin(2.0 * np.pi * x[0] / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)

# REMOVE THIS WHEN YOU SOLVE THE VARIATIONAL PROBLEM
# fsp.u.interpolate(u_expression(element=fsp.Q.ufl_element()))

fsp.v.interpolate(v_expression(element=fsp.Q.ufl_element()))
fsp.ys.interpolate(ys_expression(element=fsp.Q.ufl_element()))



'''
Here the solution of the boundary-value problem may be non unique: to make it unique one may add a Dirichlet boundary condition on the left vertex
'''
# bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.vf, rmsh.parameters['vertex_l_id'])
# bcs = [bc_u_l]

bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F = (fsp.u[alpha] - (fsp.v[beta] * bgeo.n_ale(fsp.ys, fsp.u)[beta]) * bgeo.n_ale(fsp.ys, fsp.u)[alpha]) * \
    (
        fsp.nu_u[alpha] -\
        (fsp.v[beta] * bgeo.delta_n_ale(fsp.ys, fsp.u, fsp.nu_u)[beta]) * bgeo.n_ale(fsp.ys, fsp.u)[alpha] - \
        (fsp.v[beta] * bgeo.n_ale(fsp.ys, fsp.u)[beta]) * bgeo.delta_n_ale(fsp.ys, fsp.u, fsp.nu_u)[alpha]
    ) * rmsh.dx
 
# variational functional for post-processing problem (pp) to obtain the gradient of u
# F_pp = (fsp.grad_u[alpha] - fsp.u.dx(alpha)) * fsp.nu_grad_u[alpha] * rmsh.dx
