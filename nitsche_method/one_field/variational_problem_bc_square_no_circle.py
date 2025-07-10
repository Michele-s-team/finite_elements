from fenics import *
import importlib
import numpy as np
import ufl_legacy as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import load_mesh as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

alpha = Constant(10.0)
h = CellDiameter(lmsh.mesh)


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(x[0]) ** 2 + 2 * np.sin(x[0] + x[1]) ** 4

    def value_shape(self):
        return (1,)


# class grad_u_expression(UserExpression):
#     def eval(self, values, x):
#         # values[0] = 2.0*x[0]
#         # values[1] = 4.0*x[1]
#         values[0] =  2 *(np.pi) *np.cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * np.cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* np.sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * np.sin(2 * (np.pi) * ((x[0]) + (x[1])))
#         values[1] = 2 * (np.pi) * np.cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * np.cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * np.sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * np.sin(2 * (np.pi)*  ((x[0]) + (x[1])))
#     def value_shape(self):
#         return (2,)

class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = - 2 * (np.cos(2 * x[0]) - 4 * np.cos(2 * (x[0] + x[1])) + 4 * np.cos(4 * (x[0] + x[1])))

    def value_shape(self):
        return (1,)


fsp.f.interpolate(laplacian_u_expression(element=fsp.V.ufl_element()))
fsp.u_D.interpolate(u_expression(element=fsp.V.ufl_element()))

# this is the ordinary variational functional
F_0 = dot(grad(fsp.u), grad(fsp.v)) * rmsh.dx + fsp.f * fsp.v * rmsh.dx + (- bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.v) * rmsh.ds
# this is the term that enforces the BCs with Nitche's method
F_N = ((+ alpha / h * (fsp.u - fsp.u_D)) * fsp.v - bgeo.facet_normal[i] * (fsp.v.dx(i)) * (fsp.u - fsp.u_D)) * rmsh.ds
F = F_0 + F_N
