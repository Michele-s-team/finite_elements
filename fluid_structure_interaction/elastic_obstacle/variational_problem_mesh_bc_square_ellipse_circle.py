'''
this module solves for the variables u, \dot{u} which define the state of the mesh
'''

from fenics import *
import importlib
import ufl as ufl

import elasticity as ela
import function_spaces as fsp
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class u_msh_square_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class u_msh_dot_square_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


fsp.u_msh_square.interpolate(u_msh_square_expression(element=fsp.Q_u_msh.ufl_element()))

bc_u_msh_ellipse = DirichletBC(fsp.Q_u_msh, fsp.u_el_n_on_sub_mesh_1, rmsh.boundary_ellipse)
bc_u_msh_square = DirichletBC(fsp.Q_u_msh, fsp.u_msh_square, rmsh.boundary_square)
bcs_msh = [bc_u_msh_ellipse, bc_u_msh_square]

# variational functional for the original problem
F_u = (ela.P(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.exponent), ela.mu(fsp.u_msh_n, rpam.exponent))[k, i] * (fsp.nu_u_msh[k].dx(i))) * rmsh.dx_sub_mesh[1]

fsp.u_msh_dot_square.interpolate(u_msh_dot_square_expression(element=fsp.Q_u_msh_dot.ufl_element()))

bc_u_msh_dot_ellipse = DirichletBC(fsp.Q_u_msh_dot, fsp.u_el_dot_n_on_sub_mesh_1, rmsh.boundary_ellipse)
bc_u_msh_dot_square = DirichletBC(fsp.Q_u_msh_dot, fsp.u_msh_dot_square, rmsh.boundary_square)
bcs_msh_dot = [bc_u_msh_dot_ellipse, bc_u_msh_dot_square]

F_u_dot = ( \
                      (ela.F_dot(fsp.u_msh_dot_n)[k, j] * ela.S(fsp.u_msh_n, ela.K(fsp.u_msh_n, rpam.exponent), ela.mu(fsp.u_msh_n, rpam.exponent))[j, i] \
                       + ela.F(fsp.u_msh_n)[k, j] * ela.S_dot(fsp.u_msh_n,
                                                              fsp.u_msh_dot_n,
                                                              ela.K(fsp.u_msh_n, rpam.exponent),
                                                              ela.K_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.exponent),
                                                              ela.mu(fsp.u_msh_n, rpam.exponent),
                                                              ela.mu_dot(fsp.u_msh_n, fsp.u_msh_dot_n, rpam.exponent))[j, i]) \
                      * (fsp.nu_u_msh_dot[k].dx(i))) * rmsh.dx_sub_mesh[1]
