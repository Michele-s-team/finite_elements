from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0

    def value_shape(self):
        return (1,)



fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q_u.ufl_element()))


bc_u = DirichletBC(fsp.Q_u, fsp.u_exact, rmsh.boundary)

bcs = [bc_u]

# variational functional for the original problem (poisson equation)
F = (dot(grad(fsp.u), grad(fsp.nu_u)) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds \
