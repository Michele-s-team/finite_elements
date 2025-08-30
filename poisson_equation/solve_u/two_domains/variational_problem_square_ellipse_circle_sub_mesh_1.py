from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expression for sub_mesh 1

class u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        values[0] = np.sin(2 * (np.pi) * (x[0] + x[1])) * np.cos(2 * (np.pi) * (x[0] - x[1]) ** 2)

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2.0 * x[0]
        # values[1] = 4.0 * x[1]

        # test case 2
        values[0] = 2 * (np.pi) * np.cos(2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4 * (np.pi) * (-(x[0]) + (x[1])) * sin(
            2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.sin(2 * (np.pi) * ((x[0]) + (x[1])))
        values[1] = 2 * (np.pi) * np.cos(2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4 * (np.pi) * ((x[0]) - (x[1])) * sin(
            2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.sin(2 * (np.pi) * ((x[0]) + (x[1])))

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 6.0

        # test case 2
        values[0] = 8 * (np.pi) * (-(np.pi) * (1 + 4 * (x[0] - (x[1])) ** 2) * np.cos(2 * (np.pi) * (x[0] - (x[1])) ** 2) - np.sin(2 * (np.pi) * (x[0] - (x[1])) ** 2)) * np.sin(
            2 * (np.pi) * (x[0] + (x[1])))

    def value_shape(self):
        return (1,)


fsp.u_exact[1].interpolate(u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))
fsp.grad_u[1].interpolate(grad_u_exact_sub_mesh_1_expression(element=fsp.V[1].ufl_element()))
fsp.f[1].interpolate(laplacian_u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))

# boundary conditions for sub_mesh[1]: constrain u[1] on the whole boundary of sub_mesh[1], i.e., on the ellipse and outer rectangle (lrtb)
bcs = [ \
    DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.boundary[1]['lrtb']) \
    ]

# functional for sub_mesh[1]
F = (fsp.u[1].dx(i) * fsp.nu_u[1].dx(i) + fsp.f[1] * fsp.nu_u[1]) * rmsh.dx_sub_mesh[1] \
    - bgeo.sub_mesh_facet_normal[1][i] * fsp.grad_u[1][i] * fsp.nu_u[1] * rmsh.ds_sub_mesh[1]['ds_ellipse'] \
    - bgeo.sub_mesh_facet_normal[1][i] * (fsp.u[1].dx(i)) * fsp.nu_u[1] * rmsh.ds_sub_mesh[1]['ds_lrtb']
