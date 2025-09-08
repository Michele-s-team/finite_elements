'''
this variational problem solves the Poisson equation with Dirichlet BCs

u = u_D on \partial \Omega

by imposing the BCs with Nitsche's method.
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

h = CellDiameter(lmsh.mesh)


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(x[0]) ** 2 + 2 * np.sin(x[0] + x[1]) ** 4

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = - 2 * (np.cos(2 * x[0]) - 4 * np.cos(2 * (x[0] + x[1])) + 4 * np.cos(4 * (x[0] + x[1])))

    def value_shape(self):
        return (1,)


fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))
fsp.u_D.interpolate(u_expression(element=fsp.Q.ufl_element()))

bcs = []

# this is the ordinary variational functional
F_0 = dot(grad(fsp.u), grad(fsp.nu_u)) * rmsh.dx + fsp.f * fsp.nu_u * rmsh.dx + (- bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u) * rmsh.ds
# this is the term that enforces the BCs with Nitche's method
F_N = ((+ rpam.parameters['alpha'] / h * (fsp.u - fsp.u_D)) * fsp.nu_u - bgeo.facet_normal[i] * (fsp.nu_u.dx(i)) * (fsp.u - fsp.u_D)) * rmsh.ds
F = F_0 + F_N
