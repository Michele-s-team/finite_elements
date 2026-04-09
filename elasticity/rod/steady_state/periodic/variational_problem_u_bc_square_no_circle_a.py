'''
this variational problem describes a rod in a graviataional field \vec{g}, where \vec{g} is oriented on the negative y axis and
the rod is clamped on the line x = 0
'''

from fenics import *
import importlib
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


class u_t_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


# expression for the vector g_i in the notes
class g_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = - rpam.parameters["g"]

    def value_shape(self):
        return (2,)


# expression for the density \rho_y in the notes
class rho_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters["rho"]

    def value_shape(self):
        return (2,)


fsp.u_t.interpolate(u_t_expression(element=fsp.U.ufl_element()))
fsp.g.interpolate(g_expression(element=fsp.U.ufl_element()))
fsp.rho.interpolate(rho_expression(element=fsp.R.ufl_element()))

bc_u_t = DirichletBC(fsp.U, fsp.u_t, rmsh.boundary_t)

bcs = [bc_u_t]

# natural boundary conditions are enforced here in the boundary terms
F = ( \
                - ela.P(fsp.u, rpam.parameters["K"], rpam.parameters["mu"])[i, k] * (fsp.nu_u[i].dx(k)) \
                + fsp.rho * fsp.g[i] * fsp.nu_u[i] \
        ) * rmsh.dx \
    + bgeo.facet_normal[k] * ela.P(fsp.u, rpam.parameters["K"], rpam.parameters["mu"])[0, k] * fsp.nu_u[0] * rmsh.ds_l \
    + bgeo.facet_normal[k] * ela.P(fsp.u, rpam.parameters["K"], rpam.parameters["mu"])[0, k] * fsp.nu_u[0] * rmsh.ds_r \
    + bgeo.facet_normal[k] * ela.P(fsp.u, rpam.parameters["K"], rpam.parameters["mu"])[i, k] * fsp.nu_u[i] * rmsh.ds_t
