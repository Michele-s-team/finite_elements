'''
    Here the constraint is
    z - u = g,
    This is equivalent to solving the  following PDE for z:

    Nabla Nabla \partial_i ((2*z-g) \partial_i (2*z-g)) = f
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class z_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0

    def value_shape(self):
        return (1,)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 4 - x[1] ** 4) / 48.0

    def value_shape(self):
        return (1,)


class omega_z_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 3) / 12.0
        values[1] = (x[1] ** 3) / 12.0

    def value_shape(self):
        return (2,)


class omega_u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 3) / 12.0
        values[1] = -(x[1] ** 3) / 12.0

    def value_shape(self):
        return (2,)


class mu_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 7.0 / 144.0 * x[0] ** 6

    def value_shape(self):
        return (1,)


class f_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 35.0 / 24.0 * x[0] ** 4

    def value_shape(self):
        return (1,)


class g_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 0
        values[1] = x[1] ** 3 / 6.0

    def value_shape(self):
        return (2,)


# initial profiles for the solver
class z_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0
        z_exact_expression().eval(values, x)

    def value_shape(self):
        return (1,)


class u_0_expression(UserExpression):
    def eval(self, values, x):
        # u_exact_expression().eval(values, x)
        values[0] = 0

    def value_shape(self):
        return (1,)


class omega_z_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = (x[0] ** 3) / 6.0
        # values[1] = 0.0
        omega_z_exact_expression().eval(values, x)

    def value_shape(self):
        return (2,)


class omega_u_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = (x[0] ** 3) / 6.0
        # values[1] = 0.0
        omega_u_exact_expression().eval(values, x)

    def value_shape(self):
        return (2,)


class mu_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 0
        mu_exact_expression().eval(values, x)

    def value_shape(self):
        return (1,)


fsp.z_exact.interpolate(z_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.omega_z_exact.interpolate(omega_z_exact_expression(element=fsp.Q_omega_z.ufl_element()))
fsp.omega_u_exact.interpolate(omega_u_exact_expression(element=fsp.Q_omega_u.ufl_element()))
fsp.mu_exact.interpolate(mu_exact_expression(element=fsp.Q_mu.ufl_element()))
fsp.f.interpolate(f_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.g.interpolate(g_exact_expression(element=fsp.Q_omega_z.ufl_element()))

fsp.z_0.interpolate(z_0_expression(element=fsp.Q_z.ufl_element()))
fsp.u_0.interpolate(u_0_expression(element=fsp.Q_u.ufl_element()))
fsp.omega_z_0.interpolate(omega_z_exact_expression(element=fsp.Q_omega_z.ufl_element()))
fsp.omega_u_0.interpolate(omega_u_exact_expression(element=fsp.Q_omega_u.ufl_element()))
fsp.mu_0.interpolate(mu_0_expression(element=fsp.Q_mu.ufl_element()))

# boundary conditions for fourth-order PDE
bc_z = DirichletBC(fsp.Q.sub(0), fsp.z_exact, rmsh.boundary)
bc_mu = DirichletBC(fsp.Q.sub(4), fsp.mu_exact, rmsh.boundary)

# boundary condition for the constraint
bc_u = DirichletBC(fsp.Q.sub(1), fsp.u_exact, rmsh.boundary_r)

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.u_0, fsp.omega_z_0, fsp.omega_u_0, fsp.mu_0])

F_z = ((fsp.mu.dx(j)) * (fsp.nu_z.dx(j)) + fsp.f * fsp.nu_z) * rmsh.dx \
      - bgeo.facet_normal[j] * (fsp.mu.dx(j)) * fsp.nu_z * rmsh.ds

F_constraint = ((fsp.z - fsp.u).dx(i) - fsp.g[i]) * ((fsp.nu_z - fsp.nu_u).dx(i)) * rmsh.dx

F_omega_z = (fsp.z * ((fsp.nu_omega_z[i]).dx(i)) + fsp.omega_z[i] * fsp.nu_omega_z[i]) * rmsh.dx \
            - bgeo.facet_normal[i] * fsp.z * fsp.nu_omega_z[i] * rmsh.ds

F_omega_u = (fsp.u * ((fsp.nu_omega_u[i]).dx(i)) + fsp.omega_u[i] * fsp.nu_omega_u[i]) * rmsh.dx \
            - bgeo.facet_normal[i] * fsp.u * fsp.nu_omega_u[i] * rmsh.ds

F_mu = ((fsp.z + fsp.u) * (fsp.omega_z[i] + fsp.omega_u[i]) * (fsp.nu_mu.dx(i)) + fsp.mu * fsp.nu_mu) * rmsh.dx \
       - bgeo.facet_normal[i] * (fsp.z + fsp.u) * (fsp.omega_z[i] + fsp.omega_u[i]) * fsp.nu_mu * rmsh.ds

F_N = rpam.parameters['alpha'] / rmsh.r_mesh * ((fsp.z - fsp.u).dx(i) - fsp.g[i]) * ((fsp.nu_z - fsp.nu_u).dx(i)) * rmsh.ds

F = (F_z + F_constraint + F_omega_z + F_omega_u + F_mu) + F_N
bcs = [bc_z, bc_u, bc_mu]
