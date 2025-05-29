from fenics import *
import importlib
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import boundary_geometry as bgeo
import elasticity as ela
import geometry as geo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


#CHANGE PARAMETERS HERE
exponent = 3
#CHANGE PARAMETERS HERE

class u_in_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (1,)


class u_out_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (1,)


fsp.u_in.interpolate(u_in_expression(element=fsp.U.ufl_element()))
fsp.u_out.interpolate(u_out_expression(element=fsp.U.ufl_element()))

bc_u_in = DirichletBC(fsp.U, fsp.u_in, rmsh.boundary_ellipse)
bc_u_out = DirichletBC(fsp.U, fsp.u_out, rmsh.boundary_square)
bcs = [bc_u_in, bc_u_out]

# variational functional for the original problem
F = (ela.F(fsp.u)[k, j] * ela.S(fsp.u, ela.K(fsp.u, exponent), ela.mu(fsp.u, exponent))[j, i] * (fsp.nu_u[k].dx(i))) * rmsh.dx

# variational functional for post-processing problem (pp)
# F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
#        - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
