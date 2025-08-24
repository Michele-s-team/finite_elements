from fenics import *
import importlib
import numpy as np
import ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 1 + x[0] ** 2

        # test case 2
        values[0] = 1 + np.cos(2 * np.pi * x[0]) / (1 + x[0] ** 2)

    def value_shape(self):
        return (1,)


class v_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2 * x[0]

        # test case 2
        values[0] = -((2 * (x[0] * np.cos(2 * np.pi * x[0]) + np.pi * (1 + x[0] ** 2) * np.sin(2 * np.pi * x[0]))) / (1 + x[0] ** 2) ** 2)

    def value_shape(self):
        return (1,)


class laplacian_u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2

        # test case 2
        values[0] = (-2 * (1 - 3 * x[0] ** 2 + 2 * np.pi ** 2 * (1 + x[0] ** 2) ** 2) * np.cos(2 * np.pi * x[0]) + 8 * np.pi * x[0] * (1 + x[0] ** 2) * np.sin(2 * np.pi * x[0])) / (1 + x[0] ** 2) ** 3

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.v_exact.interpolate(v_exact_expression(element=fsp.Q_v.ufl_element()))
fsp.laplacian_u_exact.interpolate(laplacian_u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.f.interpolate(laplacian_u_exact_expression(element=fsp.Q_u.ufl_element()))

# define Difichlet boundary conditions
bc_u = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary)
bcs = [bc_u]

# define variational problem
F_v = (fsp.v[i] * fsp.nu_v[i] + fsp.u * (fsp.nu_v[i].dx(i))) * rmsh.dx \
      - bgeo.facet_normal[i] * fsp.u * fsp.nu_v[i] * rmsh.ds
F_u = (fsp.v[i] * (fsp.nu_u.dx(i)) + fsp.f * fsp.nu_u) * rmsh.dx \
      - bgeo.facet_normal[i] * fsp.v[i] * fsp.nu_u * rmsh.ds
F_N = rpam.parameters['alpha'] / rmsh.r_mesh * (bgeo.facet_normal[i] * fsp.v[i] - bgeo.facet_normal[i] * fsp.v_exact[i]) * bgeo.facet_normal[j] * fsp.nu_v[j] * rmsh.ds

F = F_u + F_v + F_N
