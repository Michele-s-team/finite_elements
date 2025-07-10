from fenics import *
import importlib
import ufl_legacy as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

eta = 10.0


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0 + (x[0]) ** 2 + 2.0 * ((x[1]) ** 2)
        values[1] = 1.0 + (x[0]) ** 2 - 2.0 * ((x[1]) ** 2)

    def value_shape(self):
        return (2,)


class grad_u0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

    def value_shape(self):
        return (4,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 6.0
        values[1] = -2.0

    def value_shape(self):
        return (2,)


fsp.f.interpolate(laplacian_u_expression(element=fsp.V.ufl_element()))
fsp.g.interpolate(u_expression(element=fsp.V.ufl_element()))
fsp.grad_u_0.interpolate(grad_u0_expression(element=fsp.V.ufl_element()))

F_0 = (((fsp.u[i]).dx(j)) * ((fsp.v[i]).dx(j)) + fsp.f[i] * fsp.v[i]) * rmsh.dx - ((bgeo.facet_normal[i] * fsp.grad_u_0[i]) * fsp.v[0] + bgeo.facet_normal[i] * ((fsp.u[1]).dx(i)) * fsp.v[1]) * rmsh.ds
F_N = (eta * (bgeo.facet_normal[j] * fsp.u[j] - bgeo.facet_normal[j] * fsp.g[j]) * (fsp.v[i] * bgeo.facet_normal[i])) * rmsh.ds
F = F_0 + F_N
