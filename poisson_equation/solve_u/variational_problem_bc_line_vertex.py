'''
This variational problem solves for a Poisson problem on a line with a vertex in between, where the solution in the left and right halves are  mirror of each other

'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.load as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = np.cos(2 * np.pi * x[0]/ (rmsh.parameters['x_r'] - rmsh.parameters['x_l'])) 

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = - 2 * np.pi / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2 * np.pi * x[0]/ (rmsh.parameters['x_r'] - rmsh.parameters['x_l'])) 

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] =( 2 * np.pi / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']) )**2 * np.cos(2 * np.pi * x[0]/ (rmsh.parameters['x_r'] - rmsh.parameters['x_l'])) 

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        values[0] =( 2 * np.pi / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']) )**2 * np.cos(2 * np.pi * x[0]/ (rmsh.parameters['x_r'] - rmsh.parameters['x_l'])) 

    def value_shape(self):
        return (1, 1)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))


bc_u = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary)
bcs = [bc_u]

# variational functional for the original problem (poisson equation)
F = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * fsp.grad_u[i] * fsp.nu_u * rmsh.ds

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
