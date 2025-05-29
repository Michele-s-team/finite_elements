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

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2
        values[1] = 1 + x[0] ** 2 - 2 * x[1] ** 2

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 6
        values[1] = -2

    def value_shape(self):
        return (1,)



fsp.u_exact.interpolate(u_exact_expression(element=fsp.U.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.U.ufl_element()))




bc_u = DirichletBC(fsp.U, fsp.u_exact, rmsh.boundary)
bcs = [bc_u]

# variational functional for the original problem (poisson equation)
F = (fsp.u[j].dx(i) * fsp.nu_u[j].dx(i) + fsp.f[j] * fsp.nu_u[j]) * rmsh.dx \
    - bgeo.facet_normal[i] * ((fsp.u[j]).dx(i)) * fsp.nu_u[j] * rmsh.ds_lr \
    - bgeo.facet_normal[i] * ((fsp.u[j]).dx(i)) * fsp.nu_u[j] * rmsh.ds_tb \
    - bgeo.facet_normal[i] * ((fsp.u[j]).dx(i)) * fsp.nu_u[j] * rmsh.ds_ellipse

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
# F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
#        - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
