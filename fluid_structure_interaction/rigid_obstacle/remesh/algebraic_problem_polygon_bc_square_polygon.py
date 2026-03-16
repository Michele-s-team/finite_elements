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
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import parameters.read.solution as rpam
import solution_paths as solpath
import switch_problem as swi
from calculus import atan_quad

rmsh = importlib.import_module(swi.rmsh)
vp_fluid = importlib.import_module(swi.vp_fluid)

dt = rpam.parameters["T"] / rpam.parameters["num_steps"]  # time step size

focus = cal.ellipse_focal_points(rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'])[0]



i, j, k, l, m, n = ufl.indices(6)



class ys_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s)[0]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


class dyds_ellipse_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s)[1]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


fsp.ys_ellipse.interpolate(ys_ellipse_expression(element=fsp.Q_y.ufl_element()))
fsp.dyds_ellipse.interpolate(dyds_ellipse_expression(element=fsp.Q_dyds.ufl_element()))


io.full_print(fsp.ys_ellipse, 'ys_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.mesh, 'vector')

io.full_print(fsp.dyds_ellipse, 'dyds_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.mesh, 'vector')

'''
by replacing '1' in the integrant with a function of 0 =< s < 1, integral_ellipse gives \int ds f(s)
'''

# fsp.v_n_1.interpolate(vp_fluid.v_expression(element=fsp.Q_v.ufl_element()))
# fsp.sigma_n_32.interpolate(vp_fluid.sigma_expression(element=fsp.Q_phi.ufl_element()))
# io.full_print(fsp.v_n_1, 'v_n_1', \
#               solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
#               lmsh.mesh, 'vector')

# momentum of forces exerted by the fluid on the ellipse
M_ellipse = assemble( \
    (geo.epsilon[i, j] * (fsp.ys_ellipse[i] + fsp.u_n_1[i] - (Constant(focus))[i]) * ela.var_sigma_tensor(fsp.sigma_n_32, fsp.v_n_1, fsp.u_n_1, rpam.parameters["mu"])[j, k] * geo.epsilon[k, m] * ela.F(fsp.u_n_1)[m, l] * fsp.dyds_ellipse[l]) \
    / sqrt(fsp.dyds_ellipse[n] * fsp.dyds_ellipse[n]) * rmsh.ds_poly)



# print(f'int_ellipse = {M_ellipse}')
