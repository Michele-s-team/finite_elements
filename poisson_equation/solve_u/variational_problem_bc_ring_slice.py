from fenics import *
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import read_mesh_ring_slice as rmsh

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        values[0] = np.cos(2 * np.pi * (x[0] - x[1]) / rmsh.R)

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2.0 * x[0]
        # values[1] = 4.0 * x[1]

        #     test case 2
        values[0] = -np.pi * np.sin(np.pi * (x[0] - x[1]))
        values[1] = np.pi * np.sin(np.pi * (x[0] - x[1]))

    def value_shape(self):
        return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 6.0

        # test case 2
        values[0] = -2 * np.pi ** 2 * np.cos(np.pi * (x[0] - x[1]))

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        # values[0] = 2
        # values[1] = 0
        # values[2] = 0
        # values[3] = 4

        # test case 2    
        cos_val = np.cos(np.pi * (x[0] - x[1]))
        values[0] = -np.pi ** 2 * cos_val  # [0][0]
        values[1] = np.pi ** 2 * cos_val  # [0][1]
        values[2] = np.pi ** 2 * cos_val  # [1][0]
        values[3] = -np.pi ** 2 * cos_val  # [1][1]

    def value_shape(self):
        return (2, 2)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

bc_u_tb = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_line_tb)
bcs = [bc_u_tb]

# variational functional for the original problem (poisson equation)
F = (dot(grad(fsp.u), grad(fsp.nu_u)) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * fsp.grad_u[i] * fsp.nu_u * rmsh.ds_arc_rR \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_line_tb

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
