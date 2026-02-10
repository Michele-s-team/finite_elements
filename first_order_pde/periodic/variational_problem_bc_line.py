from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = np.cos(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)


class grad_u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = - 2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = - 2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)



fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u_exact.interpolate(grad_u_exact_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(f_expression(element=fsp.Q.ufl_element()))


'''
Here the solution of the boundary-value problem may be non unique: to make it unique one may add a Dirichlet boundary condition on the left vertex
'''
bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.vf, rmsh.parameters['vertex_l_id'])
bcs = [bc_u_l]

# bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F = (fsp.u.dx(0) - fsp.f) * fsp.nu_u.dx(0) * rmsh.dx
 
# variational functional for post-processing problem (pp) to obtain the gradient of u
F_pp = (fsp.grad_u[i] - fsp.u.dx(i)) * fsp.nu_grad_u[i] * rmsh.dx
