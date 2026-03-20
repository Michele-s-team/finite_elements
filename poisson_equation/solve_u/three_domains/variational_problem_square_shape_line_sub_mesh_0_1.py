from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_sub_mesh_0_1_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = (x[0] - rmsh.lmsh.parameters['c_r'][0]) + 2 * (x[1] - rmsh.lmsh.parameters['c_r'][1])

        # test case 2
        values[0] = 2 * (x[0] - rmsh.lmsh.parameters['c_r'][0])**3 + (x[1] - rmsh.lmsh.parameters['c_r'][1])**3

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_0_1_expression(UserExpression):
    def eval(self, values, x):
 
        # test case 1
        # values[0] = 1
        # values[1] = 2

        # test case 2
        values[0] = 6 * (x[0] - rmsh.lmsh.parameters['c_r'][0])**2
        values[1] = 3 * (x[1] - rmsh.lmsh.parameters['c_r'][1])**2

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_0_1_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 0

        # test case 2
        values[0] = 12 * (x[0] - rmsh.lmsh.parameters['c_r'][0]) + 6 * (x[1] - rmsh.lmsh.parameters['c_r'][1])

    def value_shape(self):
        return (1,)


fsp.u_exact[0][1].interpolate(u_exact_sub_mesh_0_1_expression(element=fsp.Q[0][1].ufl_element()))
fsp.grad_u[0][1].interpolate(grad_u_exact_sub_mesh_0_1_expression(element=fsp.V[0][1].ufl_element()))
fsp.f[0][1].interpolate(laplacian_u_exact_sub_mesh_0_1_expression(element=fsp.Q[0][1].ufl_element()))

# boundary conditions for sub_mesh[0][1]
bcs = [ \
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["shape_id"])
    ]

# variational functional for sub_mesh[1]
F = (fsp.u[0][1].dx(i) * fsp.nu_u[0][1].dx(i) + fsp.f[0][1] * fsp.nu_u[0][1]) * rmsh.dx_sub_mesh[0][1] \
    - bgeo.sub_mesh_facet_normal[0][1][i] * (fsp.u[0][1].dx(i)) * fsp.nu_u[0][1] * rmsh.ds_sub_mesh[0][1]['ds_lrtb']
