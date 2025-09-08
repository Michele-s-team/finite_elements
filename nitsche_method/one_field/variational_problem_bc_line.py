'''
this variational problem solves a one-dimensional equation by imposing that u is constant by adding a penalty term
of the form \int dx (\partial_1 u)^2 to the original variational functional
'''

from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

h = CellDiameter(lmsh.mesh)


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['u_constant']

    def value_shape(self):
        return (1,)


bc_u = DirichletBC(fsp.Q, Constant(rpam.parameters['u_constant']), rmsh.boundary_l)
bcs = [bc_u]

# this is the term that enforces the BCs with Nitche's method
'''
this term penalizes variations of u (\partial_1 u): it is derived from the functional 
G = alpha/(2 h) \int (\partial_1 u)^2 dx
by varyinng it as follows:
\delta G = alpha/h * \int dx (\partial_1 u)  \partial_1 \delta u
'''
F_N = rpam.parameters['alpha'] / h * fsp.u.dx(i) * (fsp.nu_u.dx(i)) * rmsh.dx
F = F_N
