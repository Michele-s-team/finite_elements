'''
solve for the Poisson equation in a square (A), which represents half of an imaginary square, whose right half will be denoted by B. The right edge of A represents the symmetry axis which mirrors A into B. 
All fields in A are the mirror image of their profiles in B. 
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import numpy as np
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = np.cos(2*np.pi*x[0]/rmsh.parameters['L']) * x[1]

    def value_shape(self):
        return (1,)

class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = -((4 * (np.pi)**2 * x[1] * np.cos((2 * np.pi * x[0])/(rmsh.parameters['L'])))/(rmsh.parameters['L'])**2)

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        values[0] = -4 * np.pi**2 * x[1] * np.cos(2 * np.pi * x[0] / rmsh.parameters['L']) / rmsh.parameters['L']**2
        values[1] = -2 * np.pi * np.sin(2 * np.pi * x[0] / rmsh.parameters['L']) / rmsh.parameters['L']
        values[2] = -2 * np.pi * np.sin(2 * np.pi * x[0] / rmsh.parameters['L']) / rmsh.parameters['L']
        values[3] = 0

    def value_shape(self):
        return (2, 2)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

bc_u_tb = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_tb)
bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.boundary_l)
bcs = [bc_u_l, bc_u_tb]

# variational functional for the original problem (poisson equation)
F = (fsp.u.dx(i)*fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_l \
    - bgeo.facet_normal[1] * (fsp.u.dx(1)) * fsp.nu_u * rmsh.ds_r \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_tb \

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
