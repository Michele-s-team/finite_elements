'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import calculus as cal
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi
from calculus import atan_quad

rmsh = importlib.import_module(swi.rmsh)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

i, j, k, l, m, n, o = ufl.indices(7)


# expression for the density \rho_y of the elastic body in the notes
class rho_el_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['rho_el']

    def value_shape(self):
        return (1,)


# {y_s}_notes (parameteric curve of the ellipse)
class ys_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s)[0]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


# {d y_s / ds}_notes
class dyds_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s)[1]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


fsp.ys_ellipse.interpolate(ys_ellipse_expression(element=fsp.Q_ys.ufl_element()))
fsp.dyds_ellipse.interpolate(dyds_ellipse_expression(element=fsp.Q_dyds.ufl_element()))
fsp.rho_el.interpolate(rho_el_expression(element=fsp.Q_rho_el.ufl_element()))

bc_u_el_circle = DirichletBC(fsp.Q_el.sub(0), Constant((0, 0)), rmsh.boundary[0]['circle'])
# bc_u_el_ellipse = DirichletBC(fsp.Q_el.sub(0), Constant((0,0)), rmsh.boundary[0]['ellipse'])

bcs_el = [bc_u_el_circle]
# bcs_el = [bc_u_el_circle, bc_u_el_ellipse]

# variational functional for the original problem
# natural BC imposed here
F_el_u_dot = (
                     fsp.rho_el / dt * (fsp.u_el_dot_n[i] - fsp.u_el_dot_n_1[i]) * fsp.nu_u_el_n[i] \
                     + ela.N(fsp.u_el_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * (fsp.nu_u_el_n[i].dx(k)) \
                 ) * rmsh.dx_sub_mesh[0] \
             - bgeo.sub_mesh_facet_normal[0][k] * ela.N(fsp.u_el_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] * fsp.nu_u_el_n[i] * rmsh.ds_sub_mesh[0]['ds_circle'] \
             - (flu.sigma_ale(fsp.v_n_1_on_sub_mesh_0, fsp.sigma_n_32_on_sub_mesh_0, fsp.u_el_n, rpam.parameters['mu_fluid'])[i, j] * geo.epsilon[j, k] * ela.F(fsp.u_el_n_1)[k, l] * fsp.dyds_ellipse[l] / sqrt(fsp.dyds_ellipse[m] * fsp.dyds_ellipse[m])) * fsp.nu_u_el_n[i] * rmsh.ds_sub_mesh[0]['ds_ellipse']

F_el_u = (fsp.u_el_n[i] - fsp.u_el_n_1[i] - fsp.u_el_dot_n[i] * dt) * fsp.nu_u_el_dot_n[i] * rmsh.dx_sub_mesh[0]

F_el = F_el_u_dot + F_el_u
