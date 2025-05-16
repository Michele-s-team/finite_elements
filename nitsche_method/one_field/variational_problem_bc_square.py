from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

h = CellDiameter(mesh)


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = cos(x[0]) ** 2 + 2 * sin(x[0] + x[1]) ** 4

    def value_shape(self):
        return (1,)


# class grad_u_expression(UserExpression):
#     def eval(self, values, x):
#         # values[0] = 2.0*x[0]
#         # values[1] = 4.0*x[1]
#         values[0] =  2 *(np.pi) *cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
#         values[1] = 2 * (np.pi) * cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * sin(2 * (np.pi)*  ((x[0]) + (x[1])))
#     def value_shape(self):
#         return (2,)

class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = - 2 * (cos(2 * x[0]) - 4 * cos(2 * (x[0] + x[1])) + 4 * cos(4 * (x[0] + x[1])))

    def value_shape(self):
        return (1,)

f.interpolate(laplacian_u_expression(element=V.ufl_element()))
u_D.interpolate(u_expression(element=V.ufl_element()))

#this is the ordinary variational functional
F_0 = dot(grad(u), grad(v))*dx + f*v*dx + ( - n[i]*(u.dx(i))  * v ) * ds
#this is the term that enforces the BCs with Nitche's method
F_N = ((+ alpha / h * (u - u_D)) * v - n[i] * (v.dx( i )) * (u - u_D)) * ds
F = F_0 + F_N

