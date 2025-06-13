from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import numpy as np
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(2 * np.pi * x[0] / rmsh.L) * np.cos(2 * np.pi * x[1] / rmsh.h)

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = - (2 * np.pi / rmsh.L) * np.cos(2 * np.pi * x[1] / rmsh.h) * np.sin(2 * np.pi * x[0] / rmsh.L)
        values[1] = - (2 * np.pi / rmsh.h) * np.cos(2 * np.pi * x[0] / rmsh.L) * np.sin(2 * np.pi * x[1] / rmsh.h)

    def value_shape(self):
        return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = - (4 * np.pi ** 2 * (rmsh.h ** 2 + rmsh.L ** 2) *
                       np.cos(2 * np.pi * x[0] / rmsh.L) * np.cos(2 * np.pi * x[1] / rmsh.h)) / (rmsh.h ** 2 * rmsh.L ** 2)

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # Common precomputed terms
        cos_x = np.cos(2 * np.pi * x[0] / rmsh.L)
        cos_y = np.cos(2 * np.pi * x[1] / rmsh.h)
        sin_x = np.sin(2 * np.pi * x[0] / rmsh.L)
        sin_y = np.sin(2 * np.pi * x[1] / rmsh.h)
        pi2 = 4 * np.pi ** 2

        # Matrix components
        values[0] = - (pi2 * cos_x * cos_y) / rmsh.L ** 2  # [0, 0]
        values[1] = (pi2 * sin_x * sin_y) / (rmsh.h * rmsh.L)  # [0, 1]
        values[2] = (pi2 * sin_x * sin_y) / (rmsh.h * rmsh.L)  # [1, 0]
        values[3] = - (pi2 * cos_x * cos_y) / rmsh.h ** 2  # [1, 1]

    def value_shape(self):
        return (2, 2)


# class u_0_Expression(UserExpression):
#     def eval(self, values, x):
#         # test case 1
#         # values[0] = 1 +  2 * x[1] ** 2
#         values[0] = 1 + x[0] + 2 * x[1] ** 2
#
#     def value_shape(self):
#         return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

#
# fsp.u.interpolate(u_0_Expression(element=fsp.Q.ufl_element()))
#

bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_l)
bc_u_t = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_t)
bcs = [bc_u_l, bc_u_t]

# variational functional for the original problem (poisson equation)
F = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_lr \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_tb \
 \
    # variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
