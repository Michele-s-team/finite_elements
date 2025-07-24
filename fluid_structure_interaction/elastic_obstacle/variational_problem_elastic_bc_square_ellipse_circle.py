'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import calculus as cal
import elasticity as ela
import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import load_mesh as lmsh
import read_parameters as rpam
import solution_paths as solpath
import switch_problem as swi
from calculus import atan_quad

rmsh = importlib.import_module(swi.rmsh)
# vp_fluid = importlib.import_module(swi.vp_fluid)

dt = rpam.T / rpam.num_steps  # time step size

i, j, k, l, m, n, o = ufl.indices(7)


# expression for the deformation field of the elastic body ad the inner circular boundary of the elastic body
class u_el_circle_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


# expression for the density \rho_y of the elastic body in the notes
class rho_el_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.rho_el

    def value_shape(self):
        return (2,)


# {y_s}_notes (parameteric curve of the ellipse)
class ys_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], 0, s)[0]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


# {d y_s / ds}_notes
class dyds_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], 0, s)[1]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


fsp.ys_ellipse.interpolate(ys_ellipse_expression(element=fsp.Q_ys.ufl_element()))
fsp.dyds_ellipse.interpolate(dyds_ellipse_expression(element=fsp.Q_dyds.ufl_element()))
fsp.rho_el.interpolate(rho_el_expression(element=fsp.Q_rho_el.ufl_element()))

fsp.u_el_circle.interpolate(u_el_circle_expression(element=fsp.Q_u_el.ufl_element()))

io.full_print(fsp.ys_ellipse, 'ys_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.sub_meshes[0], 'vector')

io.full_print(fsp.dyds_ellipse, 'dyds_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.sub_meshes[0], 'vector')

'''
# momentum of forces exerted by the fluid on the ellipse
M_ellipse = assemble( \
    (geo.epsilon[i, j] * (fsp.ys_ellipse[i] + fsp.u_el_n_1[i] - (Constant(rmsh.focus[:2]))[i]) * ela.var_sigma_tensor(fsp.sigma_n_32, fsp.v_n_1, fsp.u_el_n_1, rpam.mu_fluid)[j, k] * geo.epsilon[k, m] * ela.F(fsp.u_el_n_1)[m, l] * fsp.dyds_ellipse[l]) \
    / sqrt(fsp.dyds_ellipse[n] * fsp.dyds_ellipse[n]) * rmsh.ds_ellipse)
'''

bc_u_el_circle = DirichletBC(fsp.Q_el.sub(0), fsp.u_el_circle, rmsh.boundary[0]['circle'])

bcs_el = [bc_u_el_circle]

# variational functional for the original problem
F_el_u_dot = (
                     fsp.rho_el / dt * (fsp.u_el_dot_n[i] - fsp.u_el_dot_n_1[i]) * fsp.nu_u_el_n[i] \
                     + ela.P(fsp.u_el_n, rpam.K_elastic, rpam.mu_elastic)[i, k] * (fsp.nu_u_el_n[i].dx(k)) \
                 ) * rmsh.dx_sub_mesh[0] \
             - bgeo.sub_mesh_facet_normal[0][k] * ela.P(fsp.u_el_n, rpam.K_elastic, rpam.mu_elastic)[i, k] * fsp.nu_u_el_n[i] * rmsh.ds_sub_mesh[0]['ds_circle'] \
             - bgeo.sub_mesh_facet_normal[0][k] * ela.P(fsp.u_el_n, rpam.K_elastic, rpam.mu_elastic)[i, k] * fsp.nu_u_el_n[i] * rmsh.ds_sub_mesh[0]['ds_ellipse']

F_el_u = (fsp.u_el_n[i] - fsp.u_el_n_1[i] - fsp.u_el_dot_n[i] * dt) * fsp.nu_u_el_dot_n[i] * rmsh.dx_sub_mesh[0]

F_N = rpam.alpha / rmsh.r_mesh * ( \
    # in F I PUT U_N RATHER THAN U_N_1 AS IN THE NOTES -> REVISE THIS
        ( \
                    (geo.epsilon[j, k] * fsp.dyds_ellipse[k] * ela.P(fsp.u_el_n, rpam.K_elastic, rpam.mu_elastic)[i, j] - ela.var_sigma_tensor(fsp.sigma_n_32_on_sub_mesh_0, fsp.v_n_1_on_sub_mesh_0, fsp.u_el_n, rpam.mu_fluid)[i, j] * geo.epsilon[j, k] * ela.F(fsp.u_el_n)[k, l] * fsp.dyds_ellipse[l]) \
                    * (ela.var_sigma_tensor(fsp.sigma_n_32_on_sub_mesh_0, fsp.v_n_1_on_sub_mesh_0, fsp.u_el_n, rpam.mu_fluid)[i, m] * geo.epsilon[m, n] * fsp.nu_u_el_n[n].dx(o) * fsp.dyds_ellipse[o]) \
            ) * rmsh.ds_sub_mesh[0]['ds_ellipse'] \
    )

F_el = (F_el_u_dot + F_el_u) + F_N
