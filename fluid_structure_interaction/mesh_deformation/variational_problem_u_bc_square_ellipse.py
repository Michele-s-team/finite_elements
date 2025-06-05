from fenics import *
import importlib
import numpy as np
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import elasticity as ela
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)

# CHANGE PARAMETERS HERE
exponent = 3
psi = np.pi / 10
psi_dot = -1


# CHANGE PARAMETERS HERE

class u_in_expression(UserExpression):
    def eval(self, values, x):
        x_minus_focus = np.subtract(x, rmsh.focus[:2])
        displacement = np.subtract(np.dot(cal.R(psi), x_minus_focus), x_minus_focus)

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
F = (ela.F(fsp.u)[k, j] * ela.S(fsp.u, ela.K(fsp.u, exponent), ela.mu(fsp.u, exponent))[j, i] * (fsp.nu_u[k].dx(i))) * rmsh.dx
