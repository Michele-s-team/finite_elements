from fenics import *
import importlib
import numpy as np
import ufl_legacy as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 2 * x[0]
        values[1] = 4 * x[1]
        values[2] = 6 * x[2]


def value_shape(self):
    return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 12

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        values[0] = 2
        values[1] = 0
        values[2] = 0

        values[3] = 0
        values[4] = 4
        values[5] = 0

        values[6] = 0
        values[7] = 0
        values[8] = 6

    def value_shape(self):
        return (2, 2)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

bc_u_leri = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_leri)
bc_u_tobo = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_tobo)
bcs = [bc_u_leri, bc_u_tobo]

# variational functional for the original problem (poisson equation)
F = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_leri \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_tobo \
    - bgeo.facet_normal[i] * fsp.grad_u[i] * fsp.nu_u * rmsh.ds_frba\
    - bgeo.facet_normal[i] * fsp.grad_u[i] * fsp.nu_u * rmsh.ds_sphere

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
