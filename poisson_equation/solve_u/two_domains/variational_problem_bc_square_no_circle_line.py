from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function as fu
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expression for sub_mesh 0

class u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = x[0] ** 2 + x[1] ** 2

        # test case 2
        values[0] = np.cos(2 * np.pi * (x[0] - x[1])) ** 2 + np.cos(2 * np.pi * x[0] ** 2) ** 2 * np.sin(2 * np.pi * x[0]) ** 2

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2 * x[0]
        # values[1] = 2 * x[1]

        # test case 2
        values[0] = 2 * np.pi * (np.cos(2 * np.pi * x[0] ** 2) ** 2 * np.sin(4 * np.pi * x[0]) - 2 * x[0] * np.sin(2 * np.pi * x[0]) ** 2 * np.sin(4 * np.pi * x[0] ** 2) - np.sin(4 * np.pi * (x[0] - x[1])))
        values[1] = 2 * np.pi * np.sin(4 * np.pi * (x[0] - x[1]))

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 4

        # test case 2
        values[0] = np.pi * (2 * np.pi * (2 * np.cos(4 * np.pi * x[0]) + (1 - 2 * x[0]) ** 2 * np.cos(4 * np.pi * (-1 + x[0]) * x[0]) - 8 * x[0] ** 2 * np.cos(4 * np.pi * x[0] ** 2) + (1 + 2 * x[0]) ** 2 * np.cos(4 * np.pi * x[0] * (1 + x[0])) - 8 * np.cos(4 * np.pi * (x[0] - x[1]))) + np.sin(4 * np.pi * (-1 + x[0]) * x[0]) - 2 * np.sin(4 * np.pi * x[0] ** 2) + np.sin(4 * np.pi * x[0] * (1 + x[0])))

    def value_shape(self):
        return (1,)


# exact expression for sub_mesh 1

class u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = x[0] ** 2 + (rmsh.parameters['h'])**2

        # test case 2
        values[0] = np.cos(2 * np.pi * x[0] ** 2) * np.sin(2 * np.pi * x[0])

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2.0 * x[0]

        # test case 2
        values[0] = 2 * np.pi * (np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[0] ** 2) -
                                 2 * x[0] * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[0] ** 2))
    def value_shape(self):
        return (1,)


class laplacian_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        # values[0] = 2.0

        # test case 2
        values[0] = -4 * np.pi * (np.pi * (1 + 4 * x[0] ** 2) * np.cos(2 * np.pi * x[0] ** 2) * np.sin(2 * np.pi * x[0]) + (4 * np.pi * x[0] * np.cos(2 * np.pi * x[0]) + np.sin(2 * np.pi * x[0])) * np.sin(2 * np.pi * x[0] ** 2))
    def value_shape(self):
        return (1,)

# the function v is v = u[1]**2 + cos(2 pi (x[0] - h))**2
class v_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = (fsp.u[1])(x[0]) ** 2 + (np.cos(2 * np.pi * (x[0] - rmsh.parameters['h']))) ** 2

    def value_shape(self):
        return (1,)


fsp.u_exact[0].interpolate(u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))
fsp.grad_u[0].interpolate(grad_u_exact_sub_mesh_0_expression(element=fsp.V[0].ufl_element()))
fsp.f[0].interpolate(laplacian_u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))

fsp.u_exact[1].interpolate(u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))
fsp.grad_u[1].interpolate(grad_u_exact_sub_mesh_1_expression(element=fsp.V[1].ufl_element()))
fsp.f[1].interpolate(laplacian_u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))

bcs = [None] * len(rmsh.lmsh.sub_meshes)








# variational problem
F = []

# sub_mesh[0]

# solve problem 0 by using the solution of problem 1 to specify the BCs
# set the BC at the interface between sub_mesh[0] and sub_mesh[1] according to the solution fsp.u[1] obtained above
# impose the BCs for problem on sub_mesh[0], on the t boundary of sub_mesh[0], in terms of fsp.u_1_on_0, and solve problem on sub_mesh[0]
# force reload vp to update bc[0], because u_1_on_0 has changed
fsp.v.interpolate(v_Expression(element=fsp.Q[1].ufl_element()))
# set u_1_on_0 to be equal to v = u[1]**2 + cos(2 pi (x[0] - h))**2 on the top edge of sub_mesh[1]
fsp.u_1_on_0.assign(fu.transfer_sub_mesh_to_mesh(fsp.v, fsp.Q[0], fsp.Q[1], rmsh.parameters['h']))

bcs[0] = [ \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_l_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_r_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_1_on_0, rmsh.mf_sub_mesh[0], rmsh.parameters["sub_mesh_1_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.mf_sub_mesh[0], rmsh.parameters["line_sub_mesh_0_b_id"])
    ]

F.append( \
    (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_sub_mesh[0] \
    - bgeo.sub_mesh_facet_normal[0][i] * (fsp.u[0].dx(i)) * fsp.nu_u[0] * rmsh.ds_sub_mesh[0]['ds'] \
 \
    )



# sub_mesh[1]
# boundary conditions for sub_mesh[1]: constrain u[1] on the whole boundary of sub_mesh[1], i.e., on the ellipse and outer rectangle (lrtb)
bcs[1] = [ \
    DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.boundary[1]['lr']) \
    ]

F.append( \
    (fsp.u[1].dx(i) * fsp.nu_u[1].dx(i) + fsp.f[1] * fsp.nu_u[1]) * rmsh.dx_sub_mesh[1] \
    - bgeo.sub_mesh_facet_normal[1][i] * (fsp.u[1].dx(i)) * fsp.nu_u[1] * rmsh.ds_sub_mesh[1]['ds'] \
    )
