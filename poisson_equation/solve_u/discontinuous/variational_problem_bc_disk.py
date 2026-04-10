from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2 * np.pi * (x[0] + x[1]) / rpam.parameters["R"]) * np.cos(2 * np.pi * (x[0] - x[1]) / rpam.parameters["R"])

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        # test case 2
        # pi = np.pi
        # xpy = pi * (x[0] + x[1])
        # xmy = pi * (x[0] - x[1])
        # values[0] = pi * np.cos(xmy) * np.cos(xpy) - pi * np.sin(xmy) * np.sin(xpy)  # ∂u/∂x
        # values[1] = pi * np.cos(xmy) * np.cos(xpy) + pi * np.sin(xmy) * np.sin(xpy)  # ∂u/∂y


def value_shape(self):
        return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0

        # test case 2
        # pi = np.pi
        # xpy = pi * (x[0] + x[1])
        # xmy = pi * (x[0] - x[1])
        # values[0] = -4 * pi ** 2 * np.cos(xmy) * np.sin(xpy)

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
        values[3] = 4

        # test case 2
        # pi = np.pi
        # values[0] = -2 * pi ** 2 * np.sin(2 * pi * x[0])  # ∂²u/∂x²
        # values[1] = 0  # ∂²u/∂x∂y
        # values[2] = 0  # ∂²u/∂y∂x
        # values[3] = -2 * pi ** 2 * np.sin(2 * pi * x[1])  # ∂²u/∂y²

    def value_shape(self):
        return (2, 2)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
# fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

# fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

bcs = []

# variational functional for the original problem (poisson equation)
F_0 = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

F_i = (
        - (\
            msh.average(fsp.u.dx(i))    * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] +  # consistency
            msh.average(fsp.nu_u.dx(i)) * msh.jump(fsp.u,    bgeo.facet_normal)[i]   # adjoint
        ) + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS


F_e =  (\
            - bgeo.facet_normal[i] * (fsp.u - fsp.u_exact) * fsp.nu_u.dx(i) + \
            rpam.parameters['alpha']/rmsh.r_mesh * (fsp.u - fsp.u_exact) * fsp.nu_u\
        ) * rmsh.ds


F = F_0 + F_i + F_e

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
# F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
#    - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
