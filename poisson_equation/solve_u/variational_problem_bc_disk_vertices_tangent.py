'''
this variational problem solves for the poisson equation for u, with the BC that fixes (\partial_i u) t_i  on \partial \Omega. The boundary value problem is degenerate, thus I fix the solution by pinning the solution on a vertex. 
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # values[0] = np.sin(2 * np.pi * (x[0] + x[1]) / rpam.parameters["R"]) * np.cos(2 * np.pi * (x[0] - x[1]) / rpam.parameters["R"])

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        # test case 2
        # pi = np.pi
        # xpy = pi * (x[0] + x[1])
        # xmy = pi * (x[0] - x[1])
        # values[0] = pi * np.cos(xmy) * np.cos(xpy) - pi * np.sin(xmy) * np.sin(xpy)  # ∂u/∂x
        # values[1] = pi * np.cos(xmy) * np.cos(xpy) + pi * np.sin(xmy) * np.sin(xpy)  # ∂u/∂y


def value_shape(self):
        return (2,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 6.0

        # test case 2
        # pi = np.pi
        # xpy = pi * (x[0] + x[1])
        # xmy = pi * (x[0] - x[1])
        # values[0] = -4 * pi ** 2 * np.cos(xmy) * np.sin(xpy)

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
        # pi = np.pi
        # values[0] = -2 * pi ** 2 * np.sin(2 * pi * x[0])  # ∂²u/∂x²
        # values[1] = 0  # ∂²u/∂x∂y
        # values[2] = 0  # ∂²u/∂y∂x
        # values[3] = -2 * pi ** 2 * np.sin(2 * pi * x[1])  # ∂²u/∂y²

    def value_shape(self):
        return (2, 2)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(laplacian_u_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))



'''
this method returns True of x is close to point [r, 0] (vertex_0) and False otherwise
'''
def vertex_0(x, on_boundary):
    tol = DOLFIN_EPS
    return near(x[0], rmsh.parameters['r'], tol) and near(x[1], 0, tol)

'''
this BC removes the degeneracy of the variational problem, by imposing that the solution is equal to u_exact on vertex_0
'''
bc_remove_degeneracy = DirichletBC(fsp.Q, fsp.u_exact, vertex_0, method="pointwise")
bcs=[bc_remove_degeneracy]


# variational functional for the original problem (poisson equation)
F_0 = (dot(grad(fsp.u), grad(fsp.nu_u)) + fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

F_N = rpam.parameters['alpha'] / rmsh.r_mesh * (fsp.u.dx(i) * bgeo.facet_tangent[i] - fsp.u_exact.dx(i) * bgeo.facet_tangent[i]) * (fsp.nu_u.dx(j) * bgeo.facet_tangent[j]) * rmsh.ds

F = F_0 + F_N

# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
       - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
