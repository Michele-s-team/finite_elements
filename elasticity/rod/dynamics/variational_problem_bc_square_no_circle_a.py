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

dt = rpam.parameters['T'] / rpam.parameters['num_steps']


class u_l_expression(UserExpression):
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


fsp.u_l.interpolate(u_l_expression(element=fsp.U_u_n.ufl_element()))
fsp.g.interpolate(g_expression(element=fsp.G.ufl_element()))
fsp.rho.interpolate(rho_expression(element=fsp.R.ufl_element()))

bc_u_l = DirichletBC(fsp.U.sub(0), fsp.u_l, rmsh.boundary_l)

bcs = [bc_u_l]

# variational functional for the original problem
F_u = (
              fsp.rho / dt * (fsp.v_n[i] - fsp.v_n_1[i]) * fsp.nu_u_n[i] \
              + ela.P(fsp.u_n, rpam.parameters["K"], rpam.parameters["mu"])[i, k] * (fsp.nu_u_n[i].dx(k)) \
              - fsp.rho * fsp.g[i] * fsp.nu_u_n[i] \
          ) * rmsh.dx \
      - bgeo.facet_normal[k] * ela.P(fsp.u_n, rpam.parameters["K"], rpam.parameters["mu"])[i, k] * fsp.nu_u_n[i] * rmsh.ds_l

F_v = (fsp.u_n[i] - fsp.u_n_1[i] - fsp.v_n[i] * dt) * fsp.nu_v_n[i] * rmsh.dx

F = F_u + F_v
