from fenics import *
import importlib
import numpy as np
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import boundary_geometry as bgeo
import calculus as cal
import elasticity as ela
import function_spaces as fsp
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


class u_l_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class rho_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.rho

    def value_shape(self):
        return (2,)


fsp.u_l.interpolate(u_l_expression(element=fsp.U.ufl_element()))
fsp.rho.interpolate(rho_expression(element=fsp.R.ufl_element()))

bc_u_l = DirichletBC(fsp.U, fsp.u_l, rmsh.boundary_l)

bcs = [bc_u_l]

# variational functional for the original problem
F = ( \
                -ela.F(fsp.u)[i, j] * ela.S(fsp.u, rpam.K, rpam.mu)[j, k] * (fsp.nu_u[i].dx(k)) \
                + fsp.rho * fsp.g[i] * fsp.nu_u[i] \
        ) * rmsh.dx \
    + bgeo.facet_normal[k] * ela.F(fsp.u)[i, j] * ela.S(fsp.u, rpam.K, rpam.mu)[j, k] * fsp.nu_u[i] * rmsh.ds_l
