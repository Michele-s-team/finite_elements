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

class u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = (1 + x[0] ** 2 + 2 * x[1] ** 2)**2

        # test case 2
        values[0] = (np.sin(2 * (np.pi) * (x[0] + x[1])) * np.cos(2 * (np.pi) * (x[0] - x[1]) ** 2)) ** 2

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 4 * x[0] * (1 + x[0] ** 2 + 2 * x[1] ** 2)
        # values[1] = 8 * x[1] * (1 + x[0] ** 2 + 2 * x[1] ** 2)

        # test case 2
        values[0] = (
                4 * np.pi * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])) *
                (np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.cos(2 * np.pi * (x[0] + x[1])) +
                 2 * (-x[0] + x[1]) * np.sin(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])))
        )

        values[1] = (
                4 * np.pi * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])) *
                (np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.cos(2 * np.pi * (x[0] + x[1])) +
                 2 * (x[0] - x[1]) * np.sin(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])))
        )

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 8 * x[0] ** 2 + 32 * x[1] ** 2 + 12 * (1 + x[0] ** 2 + 2 * x[1] ** 2)

        # test case 2
        values[0] = (
                16 * np.pi ** 2 * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) ** 2 * np.cos(2 * np.pi * (x[0] + x[1])) ** 2
                - 16 * np.pi ** 2 * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) ** 2 * np.sin(2 * np.pi * (x[0] + x[1])) ** 2
                - 64 * np.pi ** 2 * (x[0] - x[1]) ** 2 * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) ** 2 * np.sin(2 * np.pi * (x[0] + x[1])) ** 2
                - 16 * np.pi * np.cos(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] - x[1]) ** 2) * np.sin(2 * np.pi * (x[0] + x[1])) ** 2
                + 64 * np.pi ** 2 * (x[0] - x[1]) ** 2 * np.sin(2 * np.pi * (x[0] - x[1]) ** 2) ** 2 * np.sin(2 * np.pi * (x[0] + x[1])) ** 2
        )

    def value_shape(self):
        return (1,)


fsp.u_exact[0].interpolate(u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))
fsp.grad_u[0].interpolate(grad_u_exact_sub_mesh_0_expression(element=fsp.V[0].ufl_element()))
fsp.f[0].interpolate(laplacian_u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))

# solve problem 0 by using the solution of problem 1 to specify the BCs

# set the BC at the interface between sub_mesh[0] and sub_mesh[1] according to the solution fsp,u[1] obtained above
# project fsp.u[1] on fsp.Q[0] and write the result in fsp.u_1_on_0
fsp.u_1_on_0.assign(project((fsp.u[1]) ** 2, fsp.Q[0]))

bcs = [ \
    DirichletBC(fsp.Q[0], fsp.u_1_on_0, rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["ellipse_loop_id"]) \
    ]

# functional for sub_mesh[0]
F = (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_sub_mesh[0] \
    - bgeo.sub_mesh_facet_normal[0][i] * fsp.grad_u[0][i] * fsp.nu_u[0] * rmsh.ds_sub_mesh[0]['ds_circle'] \
    - bgeo.sub_mesh_facet_normal[0][i] * (fsp.u[0].dx(i)) * fsp.nu_u[0] * rmsh.ds_sub_mesh[0]['ds_ellipse']
