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


class omega_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 3) / 6.0
        values[1] = 0.0

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
        values[0] = x[1] ** 4 / 24.0

    def value_shape(self):
        return (1,)


# initial profiles for the solver
class z_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0
        z_exact_expression().eval(values, x)

    def value_shape(self):
        return (1,)

class u_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

class omega_0_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = (x[0] ** 3) / 6.0
        # values[1] = 0.0
        omega_exact_expression().eval(values, x)


    def value_shape(self):
        return (2,)

class mu_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        # mu_exact_expression().eval(values, x)

    def value_shape(self):
        return (1,)


fsp.z_exact.interpolate(z_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.omega_exact.interpolate(omega_exact_expression(element=fsp.Q_omega.ufl_element()))
fsp.mu_exact.interpolate(mu_exact_expression(element=fsp.Q_mu.ufl_element()))
fsp.f.interpolate(f_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.g.interpolate(g_exact_expression(element=fsp.Q_z.ufl_element()))

fsp.z_0.interpolate(z_0_expression(element=fsp.Q_z.ufl_element()))
fsp.u_0.interpolate(u_0_expression(element=fsp.Q_u.ufl_element()))
fsp.omega_0.interpolate(omega_exact_expression(element=fsp.Q_omega.ufl_element()))
fsp.mu_0.interpolate(mu_0_expression(element=fsp.Q_mu.ufl_element()))


bc_z = DirichletBC(fsp.Q.sub(0), fsp.z_exact, rmsh.boundary)
bc_mu = DirichletBC(fsp.Q.sub(3), fsp.mu_exact, rmsh.boundary)

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.u_0, fsp.omega_exact, fsp.mu_exact])

F_z = ((fsp.mu.dx(j)) * (fsp.nu_z.dx(j)) + fsp.f * fsp.nu_z) * rmsh.dx \
      - bgeo.facet_normal[j] * (fsp.mu.dx(j)) * fsp.nu_z * rmsh.ds

F_u = ((fsp.z - fsp.u - fsp.g) * (fsp.nu_z - fsp.nu_u)) * rmsh.dx

F_omega = ((fsp.z + fsp.u) * ((fsp.nu_omega[i]).dx(i)) + fsp.omega[i] * fsp.nu_omega[i]) * rmsh.dx \
          - bgeo.facet_normal[i] * (fsp.z + fsp.u) * fsp.nu_omega[i] * rmsh.ds

F_mu = ((fsp.z + fsp.u) * fsp.omega[i] * (fsp.nu_mu.dx(i)) + fsp.mu * fsp.nu_mu) * rmsh.dx \
       - bgeo.facet_normal[i] * (fsp.z + fsp.u) * fsp.omega[i] * fsp.nu_mu * rmsh.ds

F_N = rpam.parameters['alpha'] / rmsh.r_mesh * ( \
            ((fsp.z - fsp.u - fsp.g) * (fsp.nu_z - fsp.nu_u)) * rmsh.ds
)

F = (F_z + F_u + F_omega + F_mu) + F_N
bcs = [bc_z, bc_mu]
