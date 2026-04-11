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


class u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        values[0] = np.sin(2 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['r']) * np.cos(2 * np.pi * (x[0] - x[1]) / rmsh.lmsh.parameters['r'])

    def value_shape(self):
        return (1,)




class laplacian_u_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 6.0

        # test case 2
        values[0] = - (4 * np.pi/rmsh.lmsh.parameters['r'])**2 * np.sin(2 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['r']) * np.cos(2 * np.pi * (x[0] - x[1]) / rmsh.lmsh.parameters['r'])

    def value_shape(self):
        return (1,)

'''
class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):

        # test case 1
        values[0] = 2
        values[1] = 0
        values[2] = 0
        values[3] = 4

        # test case 2
        # pi = np.pi
        # values[0] = -2 * pi ** 2 * np.sin(2 * pi * x[0])  # ∂²u/∂x²
        # values[1] = 0  # ∂²u/∂x∂y
        # values[2] = 0  # ∂²u/∂y∂x
        # values[3] = -2 * pi ** 2 * np.sin(2 * pi * x[1])  # ∂²u/∂y²

    def value_shape(self):
        return (2, 2)
'''

fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

bcs = []

# variational functional for the original problem (poisson equation)
F_0 = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

F_I = (
        -  msh.average(fsp.u.dx(i))    * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS

F_E =   rpam.parameters['alpha']/rmsh.r_mesh * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds


F = F_0 + F_I + F_E
