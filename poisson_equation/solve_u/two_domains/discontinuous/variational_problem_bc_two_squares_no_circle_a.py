from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

class u_exact_l_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['jump_coefficient'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2)

    def value_shape(self):
        return (1,)


class u_exact_r_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)


class laplacian_u_l_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0 + rpam.parameters['jump_coefficient'] * 4.0

    def value_shape(self):
        return (1,)


class laplacian_u_r_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0

    def value_shape(self):
        return (1,)
    
class d_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 0

    def value_shape(self):
        return (1,)


fsp.u_exact_shape.interpolate(u_exact_l_expression(element=fsp.Q.ufl_element()))
fsp.u_exact_square.interpolate(u_exact_r_expression(element=fsp.Q.ufl_element()))

fsp.f_shape.interpolate(laplacian_u_l_expression(element=fsp.Q.ufl_element()))
fsp.f_square.interpolate(laplacian_u_r_expression(element=fsp.Q.ufl_element()))

fsp.d.interpolate(d_expression(element=fsp.Q.ufl_element()))


bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx_mesh[0]['dx'] + \
        (fsp.f_shape * fsp.nu_u) * rmsh.dx_mesh[0]['dx_shape'] + \
        (fsp.f_square * fsp.nu_u) * rmsh.dx_mesh[0]['dx_square'] +\
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']

# here I put the average for d because d is the same on both sides (it is a jump)
F_shape = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.ds_mesh[0]['dS_shape']

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh[0] * ( msh.jump(fsp.u, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] )
        ) * rmsh.ds_mesh[0]['ds_I']

F_square =   rpam.parameters['alpha']/rmsh.r_mesh[0] * (fsp.u - fsp.u_exact_square) * fsp.nu_u * rmsh.ds_mesh[0]['ds']


F = F_0 + F_I + F_shape + F_square
