from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2.0 * x[0]

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2.0

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        values[0] = 2

    def value_shape(self):
        return (1, 1)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))


print("=== DEBUG: Shapes of main objects ===")
print("u:", fsp.u.ufl_shape)
print("nu_u:", fsp.nu_u.ufl_shape)
print("grad(u):", grad(fsp.u).ufl_shape)
print("grad(nu_u):", grad(fsp.nu_u).ufl_shape)
print("hess_u:", fsp.hess_u.ufl_shape)
print("nu_hess_u:", fsp.nu_hess_u.ufl_shape)
print("div(nu_hess_u):", div(fsp.nu_hess_u).ufl_shape)
print("u.dx(j):", fsp.u.dx(j).ufl_shape)
print("nu_hess_u[i,j]:", fsp.nu_hess_u[i,j].ufl_shape)
print("facet_normal:", bgeo.facet_normal.ufl_shape)
print("f:", fsp.f.ufl_shape)
print("u_exact:", fsp.u_exact.ufl_shape)
print("grad_u:", fsp.grad_u.ufl_shape)
print("hess_u_exact:", fsp.hess_u_exact.ufl_shape)
print("=====================================")

bc_u = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary)
bcs = [bc_u]

# variational functional for the original problem (poisson equation)
F = (dot(grad(fsp.u), grad(fsp.nu_u)) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * fsp.grad_u[i] * fsp.nu_u * rmsh.dp_lr

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
# F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
#        - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.dp_lr
