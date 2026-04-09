from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = (x[0] - rmsh.lmsh.parameters['c'][0]) + 2 * (x[1] - rmsh.lmsh.parameters['c'][1])

        # test case 2
        values[0] = 2 * (x[0] - rmsh.lmsh.parameters['c'][0])**3 + (x[1] - rmsh.lmsh.parameters['c'][1])**3

    def value_shape(self):
        return (1,)




class laplacian_u_exact_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 0

        # test case 2
        values[0] = 12 * (x[0] - rmsh.lmsh.parameters['c'][0]) + 6 * (x[1] - rmsh.lmsh.parameters['c'][1])

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.f.interpolate(laplacian_u_exact_expression(element=fsp.Q.ufl_element()))

# boundary conditions for sub_mesh[0][1]
bcs = [ \
    DirichletBC(fsp.Q, fsp.u_exact, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_b_id"])
    ]

# variational functional for sub_mesh[1]
F = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u) * rmsh.dx_mesh[0]['dx'] \
    - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']
