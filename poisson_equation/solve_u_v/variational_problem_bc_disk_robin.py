'''
This variational problem solves for the poisson equation 

\partial_i \partial_i u = f in \Omega

with BC 

n_i n_j \partial_i \partial_j u + u = g on \partial \Omega
'''

from fenics import *
import importlib
import numpy as np
import ufl

import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class u_exact_expression( UserExpression ):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)

class hess_u_u_exact_expression( UserExpression ):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + 3 * x[0] ** 2 + 6 * x[1] ** 2

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        
        # test case 1
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

    def value_shape(self):
        return (2,)


class laplacian_u_exact_expression( UserExpression ):
    def eval(self, values, x):
    
        # test case 1
        values[0] = 6.0

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.hess_u_u_exact.interpolate(hess_u_u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.v_exact.interpolate(v_exact_expression(element=fsp.Q_v.ufl_element()))
fsp.laplacian_u_exact.interpolate(laplacian_u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.f.interpolate(laplacian_u_exact_expression(element=fsp.Q_u.ufl_element()))

# define Difichlet boundary conditions
# bc_u = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary)
bcs = []

#define variational problem
F_u = (fsp.v[i] * (fsp.nu_u.dx( i )) + fsp.f * fsp.nu_u) * rmsh.dx \
      - bgeo.facet_normal[i] * fsp.v[i] * fsp.nu_u * rmsh.ds
F_v = (fsp.v[i] - fsp.u.dx(i)) * (fsp.nu_v[i] - fsp.nu_u.dx(i)) * rmsh.dx 

F_N = rpam.parameters['alpha'] / rmsh.r_mesh * (bgeo.facet_normal[i] * bgeo.facet_normal[j] * (fsp.v[j]).dx(i) + fsp.u - fsp.hess_u_u_exact) *\
       (bgeo.facet_normal[k] * bgeo.facet_normal[l] * (fsp.nu_v[l]).dx(k) + fsp.nu_u) * rmsh.ds

F = F_u + F_v + F_N
