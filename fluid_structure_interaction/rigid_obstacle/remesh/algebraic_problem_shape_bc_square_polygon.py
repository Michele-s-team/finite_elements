'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import elasticity as ela
import fluid as flu
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi
from calculus import atan_quad

rmsh = importlib.import_module(swi.rmsh)
vp_fluid = importlib.import_module(swi.vp_fluid)

dt = rpam.parameters["T"] / rpam.parameters["num_steps"]  # time step size

i, j, k, l, m, n = ufl.indices(6)



class ys_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]
        values[1] = x[1]

    def value_shape(self):
        return (2,)


class dyds_shape_expression(UserExpression):
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


fsp.ys_shape.interpolate(ys_shape_expression(element=fsp.Q_y.ufl_element()))
fsp.dyds_shape.interpolate(dyds_shape_expression(element=fsp.Q_dyds.ufl_element()))


# momentum of forces exerted by the fluid on the ellipse
M_ellipse = assemble( \
    (geo.epsilon[i, j] * (fsp.ys_shape[i] + fsp.u_n_1[i] - (Constant(rmsh.parameters['c']))[i]) * flu.sigma_ale(fsp.v_n_1, fsp.sigma_n_32, fsp.u_n_1, rpam.parameters["mu"])[j, k] * geo.epsilon[k, m] * ela.F(fsp.u_n_1)[m, l] * fsp.dyds_shape[l]) \
    / sqrt(fsp.dyds_shape[n] * fsp.dyds_shape[n]) * rmsh.ds_poly)
