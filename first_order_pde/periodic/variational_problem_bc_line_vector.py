from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

alpha, beta = ufl.indices(2)
epsilon = ufl.PermutationSymbol(2)


class v_expression(UserExpression):
    def eval(self, values, x):

        values[0] = - 2*np.pi*2*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]**2/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        values[1] = - 2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (2,)

fsp.v.interpolate(v_expression(element=fsp.Q.ufl_element()))

def hat_n(u):
    V = as_tensor(-epsilon[alpha, beta] * (u.dx(0))[beta], (alpha))
    return as_tensor(V[alpha] / geo.ufl_norm(u.dx(0)), (alpha))


'''
Here the solution of the boundary-value problem may be non unique: to make it unique one may add a Dirichlet boundary condition on the left vertex
'''
# bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.vf, rmsh.parameters['vertex_l_id'])
# bcs = [bc_u_l]

bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F = (fsp.u[alpha].dx(0) - fsp.v[alpha]) * fsp.nu_u.dx(0)[alpha] * rmsh.dx
 
# variational functional for post-processing problem (pp) to obtain the gradient of u
# F_pp = (fsp.grad_u[alpha] - fsp.u.dx(alpha)) * fsp.nu_grad_u[alpha] * rmsh.dx
