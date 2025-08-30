'''
Here the constraint is
u - v = g,
This is equivalent to solving the  following PDE for u:
Nabla (2 u - g) = f
or
Nabla u = (f + Nabla g)/2


Test case 1:
    We take
    f = 2
    g = x[0]**2
    ->
    Nabla u = 2
    u - v = x[0]**2

    A solution for these equations is
    u = (x[0]**2 + x[1]**2)/2
    v = u - x[0]**2  = (-x[0]**2 + x[1]**2)/2

    This satisfies both PDES:

    u + v = (x[0]**2 + x[1]**2)/2 + (-x[0]**2 + x[1]**2)/2 = x[1]**2
    u - v = (x[0]**2 + x[1]**2)/2 - (-x[0]**2 + x[1]**2)/2 = x[0]**2

    Nabla (u + v) = 2 OK
    u - v = g OK

Test case 2:
    We take
    u = Sin[2 \[Pi] (x + y)] Cos[2 \[Pi] (x - y)^2]
    v = Cos[2 \[Pi] (x - y)] Sin[2 \[Pi] (x + y)^2]
    ->
    f = -8 * np.pi * (np.pi * (1 + 4 * (x[0] - x[1]) ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) + np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1])) + 8 * np.pi * np.cos(**
    g = np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])) - np.cos(2 * np.pi * (x[0] - x[1])) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2)


'''


from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.sin(2 * np.pi * (x[0] + x[1])) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)

    def value_shape(self):
        return (1,)


class v_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(2 * np.pi * (x[0] - x[1])) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2)

    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):
        values[0] = -8 * np.pi * (np.pi * (1 + 4 * (x[0] - x[1]) ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) + np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1])) + 8 * np.pi * np.cos(2 * np.pi * (x[0] - x[1])) * (np.cos(2 * np.pi * (x[0] + x[1]) ** 2) - np.pi * (1 + 4 * (x[0] + x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2))

    def value_shape(self):
        return (1,)


class g_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])) - np.cos(2 * np.pi * (x[0] - x[1])) * np.sin(2 * np.pi * (x[0] + x[1]) ** 2)

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.v_exact.interpolate(v_exact_expression(element=fsp.Q_v.ufl_element()))
fsp.f.interpolate(f_expression(element=fsp.Q_u.ufl_element()))
fsp.g.interpolate(g_expression(element=fsp.Q_v.ufl_element()))

# fsp.assigner.assign(fsp.psi, [fsp.u_exact, fsp.v_exact])

bc_u_r = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary_r)
bc_u_R = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary_R)

bcs = [bc_u_r, bc_u_R]

# variational functional for the original problem (poisson equation)
F_u = (((fsp.u + fsp.v).dx(i)) * ((fsp.nu_u + fsp.nu_v).dx(i)) + fsp.f * (fsp.nu_u + fsp.nu_v)) * rmsh.dx \
      - bgeo.facet_normal[i] * ((fsp.u + fsp.v).dx(i)) * (fsp.nu_u + fsp.nu_v) * rmsh.ds_r \
      - bgeo.facet_normal[i] * ((fsp.u + fsp.v).dx(i)) * (fsp.nu_u + fsp.nu_v) * rmsh.ds_R

F_v = ((fsp.u - fsp.v - fsp.g) * (fsp.nu_u - fsp.nu_v)) * rmsh.dx

# here you need to include both the test function for u and the test funciton for v to make this constraint work
F_N  = rpam.parameters['alpha'] /rmsh.r_mesh *   ((fsp.u - fsp.v - fsp.g) * (fsp.nu_u - fsp.nu_v)) * rmsh.ds

F = (F_u + F_v) + F_N
