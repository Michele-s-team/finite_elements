from fenics import *
import importlib
import numpy as np
import ufl as ufl

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
        values[0] = (np.sin(2 * (np.pi) * (x[0] + x[1])) * np.cos(2 * (np.pi) * (x[0] - x[1]) ** 2))**2


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


fsp.u_exact[0].interpolate(u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))
fsp.grad_u[0].interpolate(grad_u_exact_sub_mesh_0_expression(element=fsp.V[0].ufl_element()))
fsp.f[0].interpolate(laplacian_u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))

fsp.u_exact[1].interpolate(u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))
fsp.grad_u[1].interpolate(grad_u_exact_sub_mesh_1_expression(element=fsp.V[1].ufl_element()))
fsp.f[1].interpolate(laplacian_u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))

bcs  = [None] * len(rmsh.lmsh.sub_meshes)


# boundary conditions for sub_mesh[1]: constrain u[1] on the whole boundary of sub_mesh[1], i.e., on the ellipse and outer rectangle (lrtb)
bcs[1] = [ \
    DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.boundary[1]['ellipse']), \
    DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.boundary[1]['lrtb']) \
    ]

# sub_mesh_facet_normal = []
# for p in range(len(rmsh.lmsh.sub_meshes)):
#     sub_mesh_facet_normal.append(FacetNormal(rmsh.lmsh.sub_meshes[p]))

# variational functional
F = []

# functional for sub_mesh[0]
F.append( \
    (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_sub_mesh[0]
)
# functional for sub_mesh[1]
F.append( \
    (fsp.u[1].dx(i) * fsp.nu_u[1].dx(i) + fsp.f[1] * fsp.nu_u[1]) * rmsh.dx_sub_mesh[1]
)
