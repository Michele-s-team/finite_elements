'''
solve the poisson equation on a disk \Omega with dirichlet BCs u = g on \partial \Omega imposed with a penalty term
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2
 
    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0 / 9.0

    def value_shape(self):
        return (1,)


msh.interpolate_dg(fsp.u_exact, u_exact_expression())
msh.interpolate_dg(fsp.f, f_expression())


bcs = []

# variational functional
F_0 = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.s * fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

F_I = (
        -  msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS

F_E =   rpam.parameters['alpha']/rmsh.r_mesh * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds

F_s = (fsp.s/assemble(Constant(1.0) * rmsh.ds) - fsp.u) * fsp.nu_s * rmsh.ds


F_u = F_0 + F_I + F_E

F = F_u + F_s
