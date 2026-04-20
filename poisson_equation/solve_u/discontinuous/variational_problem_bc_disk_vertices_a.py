'''
solve the poisson equation on a disk \Omega with Neumann BCs n^i \partial_i u = g on \partial \Omega, imposed as natural BCs
'''

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
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['r']) * np.cos(2 * np.pi * (x[0] - x[1]) / rmsh.lmsh.parameters['r'])

    def value_shape(self):
        return (1,)
    

class grad_u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 2 * x[0]
        values[1] = 4 * x[1]

    def value_shape(self):
        return (2,)




class laplacian_u_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0

        # test case 2
        # values[0] = - (4 * np.pi/rmsh.lmsh.parameters['r'])**2 * np.sin(2 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['r']) * np.cos(2 * np.pi * (x[0] - x[1]) / rmsh.lmsh.parameters['r'])

    def value_shape(self):
        return (1,)


msh.interpolate_dg(fsp.u_exact, u_exact_expression(), rmsh.sf)
msh.interpolate_dg(fsp.grad_u_exact, grad_u_exact_expression(),  rmsh.sf)
msh.interpolate_dg(fsp.f, laplacian_u_expression(), rmsh.sf)


bcs = []

# variational functional for the original problem (poisson equation)
F_0 = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * fsp.grad_u_exact[i] * fsp.nu_u * rmsh.ds

F_I = (
        -  msh.average(fsp.u.dx(i))    * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS

# F_E =   rpam.parameters['alpha']/rmsh.r_mesh * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds


# F = F_0 + F_I + F_E
F = F_0 + F_I 
