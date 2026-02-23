from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expression for sub_mesh 0

class u_exact_sub_mesh_0_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] =  - 2 * rmsh.lmsh.parameters['r'] * (x[0] - rmsh.lmsh.parameters['c_r'][0]) + rmsh.lmsh.parameters['r'] * (x[1] - rmsh.lmsh.parameters['c_r'][1])

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_0_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = - 2 * rmsh.lmsh.parameters['r']
        values[1] = rmsh.lmsh.parameters['r']

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_0_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 0

    def value_shape(self):
        return (1,)


fsp.u_exact[0][0].interpolate(u_exact_sub_mesh_0_0_expression(element=fsp.Q[0][0].ufl_element()))
fsp.grad_u[0][0].interpolate(grad_u_exact_sub_mesh_0_0_expression(element=fsp.V[0][0].ufl_element()))
fsp.f[0][0].interpolate(laplacian_u_exact_sub_mesh_0_0_expression(element=fsp.Q[0][0].ufl_element()))


bcs = [ \
    DirichletBC(fsp.Q[0][0], fsp.u_1_on_0_0, rmsh.lmsh.mf_sub_meshes[0][0], rmsh.lmsh.mesh_parameters[0]["circle_id"])
    ]

# functional for sub_mesh[0]
F = (fsp.u[0][0].dx(i) * fsp.nu_u[0][0].dx(i) + fsp.f[0][0] * fsp.nu_u[0][0]) * rmsh.dx_sub_mesh[0][0] \
    - bgeo.sub_mesh_facet_normal[0][0][i] * fsp.u[0][0].dx(i) * fsp.nu_u[0][0] * rmsh.ds_sub_mesh[0][0]['ds']
