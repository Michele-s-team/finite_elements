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
vp = importlib.import_module(swi.vp)

i, j, k = ufl.indices(3)


class u_dot_in_expression(UserExpression):
    def eval(self, values, x):
        x_minus_focus = np.subtract(x, rmsh.focus[:2])
        displacement_dot = rpam.psi_dot * np.dot(cal.dRddtheta(rpam.psi), x_minus_focus)

        values[0] = displacement_dot[0]
        values[1] = displacement_dot[1]

    def value_shape(self):
        return (2,)


class u_dot_out_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


fsp.u_dot_in.interpolate(u_dot_in_expression(element=fsp.U_dot.ufl_element()))
fsp.u_dot_out.interpolate(u_dot_out_expression(element=fsp.U_dot.ufl_element()))

bc_u_dot_in = DirichletBC(fsp.U_dot, fsp.u_dot_in, rmsh.boundary_ellipse)
bc_u_dot_out = DirichletBC(fsp.U_dot, fsp.u_dot_out, rmsh.boundary_square)
bcs_dot = [bc_u_dot_in, bc_u_dot_out]

F_dot = ( \
                    (ela.F_dot(fsp.u_dot)[k, j] * ela.S(fsp.u, ela.K(fsp.u, rpam.exponent), ela.mu(fsp.u, rpam.exponent))[j, i] \
                     + ela.F(fsp.u)[k, j] * ela.S_dot(fsp.u,
                                                      fsp.u_dot,
                                                      ela.K(fsp.u, rpam.exponent),
                                                      ela.K_dot(fsp.u, fsp.u_dot, rpam.exponent),
                                                      ela.mu(fsp.u, rpam.exponent),
                                                      ela.mu_dot(fsp.u, fsp.u_dot, rpam.exponent))[j, i]) \
                    * (fsp.nu_u_dot[k].dx(i))) * rmsh.dx
