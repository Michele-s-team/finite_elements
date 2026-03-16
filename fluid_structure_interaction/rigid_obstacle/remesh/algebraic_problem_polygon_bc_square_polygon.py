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
        # s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        # compute the ellipse parameteric coordinate 's' corresdponding to the point 'x'
        s = cal.parameteric_coordinate_ellipse(x, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 
                                               phi=rmsh.parameters['phi'])

        # compute the ellipse curve y_s at 's'
        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s, 
                        phi=rmsh.parameters['phi'])[0]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


class dyds_ellipse_expression(UserExpression):
    def eval(self, values, x):
        # s = 1 / (2 * np.pi) * atan_quad([rmsh.parameters["b"] * (x[0] - rmsh.parameters["c"][0]), rmsh.parameters["a"] * (x[1] - rmsh.parameters["c"][1])])

        # compute the ellipse parameteric coordinate 's' corresdponding to the point 'x'
        s = cal.parameteric_coordinate_ellipse(x, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 
                                               phi=rmsh.parameters['phi'])
        
        # compute the derivative of the ellipse curve, d y_s / ds, at 's'
        t = cal.ellipse(rmsh.parameters["a"], rmsh.parameters["b"], rmsh.parameters["c"][:2], s,
                        phi=rmsh.parameters['phi'])[1]

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)


fsp.ys_ellipse.interpolate(ys_ellipse_expression(element=fsp.Q_y.ufl_element()))
fsp.dyds_ellipse.interpolate(dyds_ellipse_expression(element=fsp.Q_dyds.ufl_element()))


# momentum of forces exerted by the fluid on the ellipse
M_ellipse = assemble( \
    (geo.epsilon[i, j] * (fsp.ys_ellipse[i] + fsp.u_n_1[i] - (Constant(focus))[i]) * ela.var_sigma_tensor(fsp.sigma_n_32, fsp.v_n_1, fsp.u_n_1, rpam.parameters["mu"])[j, k] * geo.epsilon[k, m] * ela.F(fsp.u_n_1)[m, l] * fsp.dyds_ellipse[l]) \
    / sqrt(fsp.dyds_ellipse[n] * fsp.dyds_ellipse[n]) * rmsh.ds_poly)
