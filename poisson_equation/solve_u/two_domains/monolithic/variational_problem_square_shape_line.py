from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 1 + x[0]**2 + 2 * x[1]**2

    def value_shape(self):
        return (1,)
    
class u_exact_square_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 1 + x[0]**2 + 2 * x[1]**2

    def value_shape(self):
        return (1,)

class laplacian_u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6

    def value_shape(self):
        return (1,)
    
class laplacian_u_exact_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6

    def value_shape(self):
        return (1,)
    
class g_shape_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 0

    def value_shape(self):
        return (1,)


fsp.u_exact_shape.interpolate(u_exact_shape_expression(element=fsp.Q.ufl_element()))
fsp.u_exact_square.interpolate(u_exact_square_expression(element=fsp.Q.ufl_element()))

fsp.f_shape.interpolate(laplacian_u_exact_shape_expression(element=fsp.Q.ufl_element()))
fsp.f_square.interpolate(laplacian_u_exact_square_expression(element=fsp.Q.ufl_element()))

fsp.g_shape.interpolate(g_shape_expression(element=fsp.Q.ufl_element()))


# boundary conditions for sub_mesh[0][1]
bcs = [ \
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_b_id"])
    ]

# variational functional for sub_mesh[1]
F_sh = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f_shape * fsp.nu_u) * rmsh.dx_mesh[0]['dx_shape'] \
        - ((bgeo.facet_normal[0])('+'))[i] * (fsp.u('+').dx(i)) * fsp.nu_u('+') * rmsh.ds_mesh[0]['ds_shape']

F_sq = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f_shape * fsp.nu_u) * rmsh.dx_mesh[0]['dx_square'] \
        - ((bgeo.facet_normal[0])('-'))[i] * (fsp.u('-').dx(i)) * fsp.nu_u('-') * rmsh.ds_mesh[0]['ds_shape']\
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']


F = F_sh + F_sq