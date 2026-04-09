from fenics import *
import importlib
import numpy as np
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import physics.elasticity as ela
import function_spaces as fsp
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


class u_in_expression(UserExpression):
    def eval(self, values, x):
        x_minus_focus = np.subtract(x, rmsh.focus[:2])
        displacement = np.subtract(np.dot(cal.R(rpam.psi), x_minus_focus), x_minus_focus)

        values[0] = displacement[0]
        values[1] = displacement[1]

    def value_shape(self):
        return (2,)


class u_out_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


fsp.u_in.interpolate(u_in_expression(element=fsp.U.ufl_element()))
fsp.u_out.interpolate(u_out_expression(element=fsp.U.ufl_element()))

bc_u_in = DirichletBC(fsp.U, fsp.u_in, rmsh.boundary_ellipse)
bc_u_out = DirichletBC(fsp.U, fsp.u_out, rmsh.boundary_square)
bcs = [bc_u_in, bc_u_out]

# variational functional for the original problem
F = (ela.F(fsp.u)[k, j] * ela.S(fsp.u, ela.K(fsp.u, rpam.exponent), ela.mu(fsp.u, rpam.exponent))[j, i] * (fsp.nu_u[k].dx(i))) * rmsh.dx
