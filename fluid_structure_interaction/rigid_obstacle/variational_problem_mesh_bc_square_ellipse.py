'''
this module solves for the variables u, \dot{u} which define the state of the mesh
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import physics.elasticity as ela
import function_spaces as fsp
import numpy as np
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class u_ellipse_expression(UserExpression):
    def eval(self, values, x):
        x_minus_focus = np.subtract(x, rmsh.focus[:2])
        displacement = np.subtract(np.dot(cal.R(fsp.theta_n), x_minus_focus), x_minus_focus)

        values[0] = displacement[0]
        values[1] = displacement[1]

    def value_shape(self):
        return (2,)


class u_square_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class u_dot_ellipse_expression(UserExpression):
    def eval(self, values, x):
        x_minus_focus = np.subtract(x, rmsh.focus[:2])
        displacement_dot = fsp.omega_n * np.dot(cal.dRddtheta(fsp.theta_n), x_minus_focus)

        values[0] = displacement_dot[0]
        values[1] = displacement_dot[1]

    def value_shape(self):
        return (2,)


class u_dot_square_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


fsp.u_ellipse.interpolate(u_ellipse_expression(element=fsp.Q_u.ufl_element()))
fsp.u_square.interpolate(u_square_expression(element=fsp.Q_u.ufl_element()))

bc_u_ellipse = DirichletBC(fsp.Q_u, fsp.u_ellipse, rmsh.boundary_ellipse)
bc_u_square = DirichletBC(fsp.Q_u, fsp.u_square, rmsh.boundary_square)
bcs = [bc_u_ellipse, bc_u_square]

# variational functional for the original problem
F_u = (ela.F(fsp.u_n)[k, j] * ela.S(fsp.u_n, ela.K(fsp.u_n, rpam.parameters["exponent"]), ela.mu(fsp.u_n, rpam.parameters["exponent"]))[j, i] * (fsp.nu_u[k].dx(i))) * rmsh.dx

fsp.u_dot_ellipse.interpolate(u_dot_ellipse_expression(element=fsp.Q_u_dot.ufl_element()))
fsp.u_dot_square.interpolate(u_dot_square_expression(element=fsp.Q_u_dot.ufl_element()))

bc_u_dot_ellipse = DirichletBC(fsp.Q_u_dot, fsp.u_dot_ellipse, rmsh.boundary_ellipse)
bc_u_dot_square = DirichletBC(fsp.Q_u_dot, fsp.u_dot_square, rmsh.boundary_square)
bcs_dot = [bc_u_dot_ellipse, bc_u_dot_square]

F_u_dot = ( \
                      (ela.F_dot(fsp.u_dot_n)[k, j] * ela.S(fsp.u_n, ela.K(fsp.u_n, rpam.parameters["exponent"]), ela.mu(fsp.u_n, rpam.parameters["exponent"]))[j, i] \
                       + ela.F(fsp.u_n)[k, j] * ela.S_dot(fsp.u_n,
                                                      fsp.u_dot_n,
                                                      ela.K(fsp.u_n, rpam.parameters["exponent"]),
                                                      ela.K_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters["exponent"]),
                                                      ela.mu(fsp.u_n, rpam.parameters["exponent"]),
                                                      ela.mu_dot(fsp.u_n, fsp.u_dot_n, rpam.parameters["exponent"]))[j, i]) \
                      * (fsp.nu_u_dot[k].dx(i))) * rmsh.dx
