from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha = 1e2


i, j, k, l = ufl.indices( 4 )

assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu, Q_rho, Q_tau] )



class z_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = np.cos( x[0] + x[1] ) * np.sin( x[0] - x[1] )
        values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0

    def value_shape(self):
        return (1,)


class omega_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (x[0] ** 3) / 12.0
        values[1] = (x[1] ** 3) / 12.0

    def value_shape(self):
        return (2,)


class mu_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (7 * x[0] ** 6 + 3 * x[0] ** 4 * x[1] ** 2 + 3 * x[0] ** 2 * x[1] ** 4 + 7 * x[1] ** 6) / 576.0

    def value_shape(self):
        return (1,)


class rho_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = x[0] * (7 * x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + x[1] ** 4) / 96.0
        values[1] = x[1] * (x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 7 * x[1] ** 4) / 96.0

    def value_shape(self):
        return (2,)


class f_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = -16 * (np.cos( 4 * x[0] ) + np.cos( 4 * x[1] ) + np.sin( 2 * x[0] ) * np.sin( 2 * x[1] ))
        values[0] = 1 / 8.0 * (3 * x[0] ** 4 + x[0] ** 2 * x[1] ** 2 + 3 * x[1] ** 4)

    def value_shape(self):
        return (1,)





z_exact.interpolate( z_exact_expression( element=Q_z.ufl_element() ) )
omega_exact.interpolate( omega_exact_expression( element=Q_omega.ufl_element() ) )
mu_exact.interpolate( mu_exact_expression( element=Q_mu.ufl_element() ) )
rho_exact.interpolate( rho_exact_expression( element=Q_rho.ufl_element() ) )
tau_exact.interpolate( f_exact_expression( element=Q_tau.ufl_element() ) )
f.interpolate( f_exact_expression( element=Q_z.ufl_element() ) )

z_profile = Expression( '(pow(x[0], 4) + pow(x[1], 4)) / 48.0', element=Q.sub( 0 ).ufl_element() )
mu_profile = Expression( '(7 * pow(x[0], 6) + 3 * pow(x[0], 4) * pow(x[1], 2) + 3 * pow(x[0], 2) * pow(x[1], 4) + 7 * pow(x[1], 6))/576.0', element=Q.sub( 2 ).ufl_element() )
rho_profile = Expression(
    ('(1.0 / 96.0) * x[0] * (7.0 * pow(x[0], 4) + 2.0 * pow(x[0], 2) * pow(x[1], 2) + pow(x[1], 4))', '(1.0 / 96.0) * x[1] * (pow(x[0], 4) + 2 * pow(x[0], 2) * pow(x[1], 2) + 7 * pow(x[1], 4))'),
    element=Q.sub( 3 ).ufl_element() )
tau_profile = Expression( '(1.0 / 8.0) * (3 * pow(x[0], 4) + pow(x[0], 2) * pow(x[1], 2) + 3 * pow(x[1], 4))', element=Q.sub( 4 ).ufl_element() )

bc_z = DirichletBC( Q.sub( 0 ), z_profile, boundary )
bc_mu = DirichletBC( Q.sub( 2 ), mu_profile, boundary )
bc_rho = DirichletBC( Q.sub( 3 ), rho_profile, boundary )
bc_tau = DirichletBC( Q.sub( 4 ), tau_profile, boundary )

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
assigner.assign( psi, [f, omega_exact, mu_exact, rho_exact, tau_exact] )

F_z = ((mu.dx( j )) * (nu_z.dx( j )) + f * nu_z) * dx \
      - n[j] * (mu.dx( j )) * nu_z * ds

F_omega = (z * ((nu_omega[i]).dx( i )) + omega[i] * nu_omega[i]) * dx \
          - n[i] * z * nu_omega[i] * ds

# F_mu = ((z * omega[i]).dx(i) * nu_mu  - mu * nu_mu) * dx
F_mu = (z * omega[i] * (nu_mu.dx( i )) + mu * nu_mu) * dx \
       - n[i] * z * omega[i] * nu_mu * ds

F_rho = (mu * ((nu_rho[i]).dx( i )) + rho[i] * nu_rho[i]) * dx \
        - n[i] * mu * nu_rho[i] * ds

F_tau = ( tau  * nu_tau + rho[i] * (nu_tau.dx(i)) ) * dx \
      - n[i] * rho[i] * nu_tau * ds

F_N = alpha / r_mesh * (n[i] * omega[i] - n[i] * omega_exact[i]) * n[j] * nu_omega[j] * ds

F = (F_omega + F_z + F_mu + F_rho + F_tau) + F_N
# bcs = [bc_z]
bcs = [bc_z, bc_mu, bc_rho, bc_tau]