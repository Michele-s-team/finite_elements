from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


i, j, k, l = ufl.indices(4)

assigner = FunctionAssigner(fsp.Q, [fsp.Q_z, fsp.Q_omega, fsp.Q_mu, fsp.Q_rho, fsp.Q_tau])


class z_exact_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = np.cos( x[0] + x[1] ) * np.sin( x[0] - x[1] )
        values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0

    def value_shape(self):
        return (1,)


class omega_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = (x[0] ** 3) / 12.0
        values[1] = (x[1] ** 3) / 12.0

    def value_shape(self):
        return (2,)


class mu_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = (7 * x[0] ** 6 + 3 * x[0] ** 4 * x[1] ** 2 + 3 * x[0] ** 2 * x[1] ** 4 + 7 * x[1] ** 6) / 576.0

    def value_shape(self):
        return (1,)


class rho_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0] * (7 * x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + x[1] ** 4) / 96.0
        values[1] = x[1] * (x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 7 * x[1] ** 4) / 96.0

    def value_shape(self):
        return (2,)


class f_exact_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = -16 * (np.cos( 4 * x[0] ) + np.cos( 4 * x[1] ) + np.sin( 2 * x[0] ) * np.sin( 2 * x[1] ))
        values[0] = 1 / 8.0 * (3 * x[0] ** 4 + x[0] ** 2 * x[1] ** 2 + 3 * x[1] ** 4)

    def value_shape(self):
        return (1,)


fsp.z_exact.interpolate(z_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.omega_exact.interpolate(omega_exact_expression(element=fsp.Q_omega.ufl_element()))
fsp.mu_exact.interpolate(mu_exact_expression(element=fsp.Q_mu.ufl_element()))
fsp.rho_exact.interpolate(rho_exact_expression(element=fsp.Q_rho.ufl_element()))
fsp.tau_exact.interpolate(f_exact_expression(element=fsp.Q_tau.ufl_element()))
fsp.f.interpolate(f_exact_expression(element=fsp.Q_z.ufl_element()))

z_profile = Expression('(pow(x[0], 4) + pow(x[1], 4)) / 48.0', element=fsp.Q.sub(0).ufl_element())

bc_z = DirichletBC(fsp.Q.sub(0), z_profile, rmsh.boundary)

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
assigner.assign(fsp.psi, [fsp.f, fsp.omega_exact, fsp.mu_exact, fsp.rho_exact, fsp.tau_exact])

F_z = ((fsp.mu.dx(j)) * (fsp.nu_z.dx(j)) + fsp.f * fsp.nu_z) * rmsh.dx \
      - bgeo.facet_normal[j] * (fsp.mu.dx(j)) * fsp.nu_z * rmsh.ds

F_omega = (fsp.z * ((fsp.nu_omega[i]).dx(i)) + fsp.omega[i] * fsp.nu_omega[i]) * rmsh.dx \
          - bgeo.facet_normal[i] * fsp.z * fsp.nu_omega[i] * rmsh.ds

# F_mu = ((fsp.z * fsp.omega[i]).dx(i) * fsp.nu_mu  - mu * fsp.nu_mu) * rmsh.dx
F_mu = (fsp.z * fsp.omega[i] * (fsp.nu_mu.dx(i)) + fsp.mu * fsp.nu_mu) * rmsh.dx \
       - bgeo.facet_normal[i] * fsp.z * fsp.omega[i] * fsp.nu_mu * rmsh.ds

F_rho = (fsp.mu * ((fsp.nu_rho[i]).dx(i)) + fsp.rho[i] * fsp.nu_rho[i]) * rmsh.dx \
        - bgeo.facet_normal[i] * fsp.mu * fsp.nu_rho[i] * rmsh.ds

F_tau = (fsp.tau * fsp.nu_tau + fsp.rho[i] * (fsp.nu_tau.dx(i))) * rmsh.dx \
        - bgeo.facet_normal[i] * fsp.rho[i] * fsp.nu_tau * rmsh.ds

F_N = rpam.parameters['alpha'] / rmsh.r_mesh * ( \
            (bgeo.facet_normal[i] * fsp.omega[i] - bgeo.facet_normal[i] * fsp.omega_exact[i]) * bgeo.facet_normal[j] * fsp.nu_omega[j] * rmsh.ds \
 \
            + (fsp.mu - ((fsp.z * fsp.omega[i]).dx(i))) * fsp.nu_mu * rmsh.ds \
            + (fsp.rho[i] - (fsp.mu.dx(i))) * fsp.nu_rho[i] * rmsh.ds \
            + (fsp.tau - ((fsp.rho[i]).dx(i))) * fsp.nu_tau * rmsh.ds \
    )

F = (F_omega + F_z + F_mu + F_rho + F_tau) + F_N
bcs = [bc_z]
