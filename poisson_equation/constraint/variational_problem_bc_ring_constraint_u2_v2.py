'''
Here the constraint is
u**2 - v**2 = g,
Solving the constraint for v, we obtain:
v = +- sqrt(u**2-g)
And replacing into the PDE, we obtain a PDE for u:

f = Nabla (u + v) = Nabla(u +- sqrt(u**2-g))


Test case 1:
    We take
    u = 1 + x[0]**2 + 2 * x[1]**2
    v = 1 + x[0]**2 - 2 * x[1]**2
    ->

    u + v = 2 + 2*x[0]**2
    f = 4
    g = 8 * ( 1+ x[0]**2)* x[1]**2

Test case 2:
    u[x_, y_] := Sin[2 \[Pi] (x + y)] Cos[2 \[Pi] (x - y)^2]
    v[x_, y_] := Cos[2 \[Pi] (x - y)] Sin[2 \[Pi] (x + y)^2]
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import read_parameters_solve as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2 * np.pi * (x[0] + x[1])) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)

    def value_shape(self):
        return (1,)


class v_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2 - 2 * x[1] ** 2

        # test case 2
        # values[0] = np.cos(2 * np.pi * (x[0] - x[1])) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2)

    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 4

        # test case 2
        # values[0] = (-8 * np.pi * (np.pi * (1 + 4 * (x[0] - x[1]) ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) +
        #                            np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1])) +
        #              8 * np.pi * np.cos(2 * np.pi * (x[0] - x[1])) * (np.cos(2 * np.pi * (x[0] + x[1]) ** 2) -
        #                                                               np.pi * (1 + 4 * (x[0] + x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2)))

    def value_shape(self):
        return (1,)


class g_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 8 * (1 + x[0] ** 2) * x[1] ** 2

        # test case 2
        # values[0] = (np.cos(2 * np.pi * (x[0] - x[1]) ** 2) ** 2 * np.sin(2 * np.pi * (x[0] + x[1])) ** 2 -
        #              np.cos(2 * np.pi * (x[0] - x[1])) ** 2 * np.sin(2 * np.pi * (x[0] + x[1]) ** 2) ** 2)

    def value_shape(self):
        return (1,)


class u_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2 + 2.001 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2.001 * np.pi * (x[0] + x[1])) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)

    def value_shape(self):
        return (1,)


class v_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + 1.01 * x[0] ** 2 - 2 * x[1] ** 2

        # test case 2
        # values[0] = np.cos(2.001 * np.pi * (x[0] - x[1])) * np.sin(2.001 * np.pi * (x[0] + x[1]) ** 2)

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.v_exact.interpolate(v_exact_expression(element=fsp.Q_v.ufl_element()))
fsp.f.interpolate(f_expression(element=fsp.Q_u.ufl_element()))
fsp.g.interpolate(g_expression(element=fsp.Q_v.ufl_element()))

# uncomment this if you want to start from the initial configuration u_0, v_0
#
print('Setting initial profiles ... ')
fsp.u_0.interpolate(u_0_expression(element=fsp.Q_u.ufl_element()))
fsp.v_0.interpolate(v_0_expression(element=fsp.Q_v.ufl_element()))
fsp.assigner.assign(fsp.psi, [fsp.u_0, fsp.v_0])
print('... done.')
#

bc_u_r = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary_r)
bc_u_R = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary_R)

bcs = [bc_u_r, bc_u_R]

# variational functional for the original problem (poisson equation)
F_u = (((fsp.u + fsp.v).dx(i)) * ((fsp.nu_u + fsp.nu_v).dx(i)) + fsp.f * (fsp.nu_u + fsp.nu_v)) * rmsh.dx \
      - bgeo.facet_normal[i] * ((fsp.u + fsp.v).dx(i)) * (fsp.nu_u + fsp.nu_v) * rmsh.ds_r \
      - bgeo.facet_normal[i] * ((fsp.u + fsp.v).dx(i)) * (fsp.nu_u + fsp.nu_v) * rmsh.ds_R

'''
F_v and F_N implement the constraint u^2-v^2=g. They are obtained by considering a functional 
G[u,v] = alpha/rmsh \int_Omega dx (u^2-v^2-g)^2
which is minimized and forces the minimization to enforce the constraint, which is satisfied at the minimum only. 

by varying G[u, v] with respect to u and v and setting \delta u = nu_u, \delta_v = nu_v, we obtain F_v and F_N
'''

F_v = rpam.parameters['alpha'] / rmsh.r_mesh * (((fsp.u) ** 2 - (fsp.v) ** 2 - fsp.g) * (2 * fsp.u * fsp.nu_u - 2 * fsp.v * fsp.nu_v)) * rmsh.dx

# here you need to include both the test function for u and the test funciton for v to make this constraint work
F_N = rpam.parameters['alpha'] / rmsh.r_mesh * (((fsp.u) ** 2 - (fsp.v) ** 2 - fsp.g) * (2 * fsp.u * fsp.nu_u - 2 * fsp.v * fsp.nu_v)) * rmsh.ds

F = (F_u + F_v) + F_N
