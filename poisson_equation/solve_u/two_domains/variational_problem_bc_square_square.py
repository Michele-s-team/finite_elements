from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import load_mesh as lmsh
import switch_problem as swi
from load_mesh import sub_meshes

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2 * (np.pi) * (x[0] + x[1])) * np.cos(2 * (np.pi) * (x[0] - x[1]) ** 2)

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        # test case 2
        # values[0] = 2 * (np.pi) * np.cos(2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4 * (np.pi) * (-(x[0]) + (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
        # values[1] = 2 * (np.pi) * np.cos(2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * np.cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4 * (np.pi) * ((x[0]) - (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))

    def value_shape(self):
        return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 6.0

        # test case 2
        # values[0] = 8 * (np.pi) * (-(np.pi) * (1 + 4 * (x[0] - (x[1])) ** 2) * np.cos(2 * (np.pi) * (x[0] - (x[1])) ** 2) - np.sin(2 * (np.pi) * (x[0] - (x[1])) ** 2)) * np.sin(
        #     2 * (np.pi) * (x[0] + (x[1])))

    def value_shape(self):
        return (1,)


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
        # values[0] = 4 * np.pi * (
        #         -4 * np.pi * (x[0] - x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1]) ** 2)
        #         - (np.pi * (1 + 4 * x[0] ** 2 - 8 * x[0] * x[1] + 4 * x[1] ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)
        #            + np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        # )
        #
        # values[1] = 4 * np.pi * (
        #         (np.pi * (-1 + 4 * x[0] ** 2 - 8 * x[0] * x[1] + 4 * x[1] ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)
        #          + np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        # )
        #
        # values[2] = values[1]
        #
        # values[3] = 4 * np.pi * (
        #         4 * np.pi * (x[0] - x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1]) ** 2)
        #         - (np.pi * (1 + 4 * x[0] ** 2 - 8 * x[0] * x[1] + 4 * x[1] ** 2) * np.cos(2 * np.pi * (x[0] - x[1]) ** 2)
        #            + np.sin(2 * np.pi * (x[0] - x[1]) ** 2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        # )

    def value_shape(self):
        return (2, 2)


for i in range(len(lmsh.sub_meshes)):
    fsp.u_exact[i].interpolate(u_exact_expression(element=fsp.Q[i].ufl_element()))
    fsp.grad_u[i].interpolate(grad_u_expression(element=fsp.V[i].ufl_element()))
    fsp.f[i].interpolate(laplacian_u_expression(element=fsp.Q[i].ufl_element()))

    fsp.hess_u_exact[i].interpolate(hess_u_exact_expression(element=fsp.T[i].ufl_element()))

bc_tb, bc_lr = [], []

for i in range(len(lmsh.sub_meshes)):
    bc_tb.append(DirichletBC(fsp.Q[i], fsp.u_exact[i], rmsh.boundary_tb[i]))
    bc_lr = DirichletBC(fsp.Q[i], fsp.u_exact[i], rmsh.boundary_lr[i])

bcs = [[bc_tb, bc_lr] for i in range(len(lmsh.sub_meshes))]

sub_mesh_facet_normal = []
for i in range(len(lmsh.sub_meshes)):
    sub_mesh_facet_normal.append(FacetNormal(lmsh.sub_meshes[i]))

# variational functional for the original problem (poisson equation)
# here the + and - are used to select the inner or outer par of quantities on interiors ds only
F = []
for p in range(len(lmsh.sub_meshes)):
    F.append( \
        (dot(grad(fsp.u[p]), grad(fsp.nu_u[p])) + fsp.f[p] * fsp.nu_u[p]) * rmsh.dx_sub_mesh[p] \
        - sub_mesh_facet_normal[p][i] * fsp.grad_u[p][i] * fsp.nu_u[p] * rmsh.ds_sub_mesh_lr \
        - sub_mesh_facet_normal[p][i] * (fsp.u[p].dx(i)) * fsp.nu_u[p] * rmsh.ds_sub_mesh_tb \
        - sub_mesh_facet_normal[p][i] * (fsp.u[p].dx(i)) * fsp.nu_u[p] * rmsh.ds_sub_mesh_lr \
        - sub_mesh_facet_normal[p][i] * fsp.grad_u[p][i] * fsp.nu_u[p] * rmsh.ds_sub_mesh_tb \
        )

# # variational functional for post-processing problem (pp) to obtain the hessian (hess)
# F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx_out \
#        - (sub_mesh_out_facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds_out\
#        - (sub_mesh_out_facet_normal('+')[i] * (fsp.u.dx(j))('+') * fsp.nu_hess_u('+')[i, j]) * rmsh.ds_in
