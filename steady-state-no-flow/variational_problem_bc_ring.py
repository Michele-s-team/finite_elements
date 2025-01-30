from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import read_mesh_ring as rmsh

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
# bending rigidity
kappa = 1.0
# density
rho = 1.0
# viscosity
eta = 1.0
C = 0.1
# values of z at the boundaries
'''
if you compare with the solution from check-with-analytical-solution-bc-ring.nb:
    - z_r(R)_const_{here} <-> zRmin(max)_{check-with-analytical-solution-bc-ring.nb}
    - zp_r(R)_const_{here} <-> zpRmin(max)_{check-with-analytical-solution-bc-ring.nb}
'''
z_r_const = C * rmsh.r
z_R_const = C * rmsh.R
zp_r_const = C
zp_R_const = C
omega_r_const = - (rmsh.r) * zp_r_const / np.sqrt( (rmsh.r) ** 2 * (1.0 + zp_r_const ** 2) )
omega_R_const = (rmsh.R) * zp_R_const / np.sqrt( (rmsh.R) ** 2 * (1.0 + zp_R_const ** 2) )
# Nitche's parameter
alpha = 1e1


class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = (2.0 + C ** 2) * kappa / (2.0 * (1.0 + C ** 2) * (geo.my_norm( x ) ** 2))
        # values[0] =  cos(2.0*(np.pi)*geo.my_norm(x))

    def value_shape(self):
        return (1,)


class z_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = C * geo.my_norm( x )

    def value_shape(self):
        return (1,)


class omega_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = C * x[0] / (geo.my_norm( x ))
        values[1] = C * x[1] / (geo.my_norm( x ))

    def value_shape(self):
        return (2,)


class mu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = C / (2.0 * sqrt( 1.0 + C ** 2 ) * geo.my_norm( x ))

    def value_shape(self):
        return (1,)


class nu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = - (C * x[0]) / (2.0 * np.sqrt( 1.0 + C ** 2 ) * (x[0] ** 2 + x[1] ** 2) ** (3.0 / 2.0))
        values[1] = - (C * x[1]) / (2.0 * np.sqrt( 1.0 + C ** 2 ) * (x[0] ** 2 + x[1] ** 2) ** (3.0 / 2.0))

    def value_shape(self):
        return (2,)


class tau_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = C / (2.0 * ((1.0 + C ** 2) * (geo.my_norm( x )) ** 2) ** (3.0 / 2.0))

    def value_shape(self):
        return (1,)


class z_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = z_r_const

    def value_shape(self):
        return (1,)


class z_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = z_R_const

    def value_shape(self):
        return (1,)


class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_r_const

    def value_shape(self):
        return (1,)


class omega_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_R_const

    def value_shape(self):
        return (1,)


# CHANGE PARAMETERS HERE


# values of z on ds_r and ds_R, to be used to check if the boundary conditions (BCs) are satisfied
z_r = interpolate( z_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

# values of \partial_i z = omega_i on the ds_r and ds_R, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
fsp.z_0.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_0.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_0.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )
fsp.nu_0.interpolate( nu_exact_Expression( element=fsp.Q_nu.ufl_element() ) )
fsp.tau_0.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

fsp.z_exact.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_exact.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_exact.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )
fsp.nu_exact.interpolate( nu_exact_Expression( element=fsp.Q_nu.ufl_element() ) )
fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0, fsp.nu_0, fsp.tau_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
z_profile = Expression( ' C * sqrt(pow(x[0], 2) + pow(x[1], 2))', C=C, element=fsp.Q.sub( 0 ).ufl_element() )
# mu_profile = Expression( ' C / (2.0 * sqrt(1.0 + pow(C, 2)) * sqrt(pow(x[0], 2) + pow(x[1], 2)))', C=C, element=fsp.Q.sub( 2 ).ufl_element() )
# nu_profile = Expression( ('- (C * x[0]) / (2.0 * sqrt(1.0 + C * C) * pow(x[0] * x[0] + x[1] * x[1], 3.0/2.0))', '- (C * x[1]) / (2.0 * sqrt(1.0 + C * C) * pow(x[0] * x[0] + x[1] * x[1], 3.0/2.0))'),
#                          C=C, element=fsp.Q.sub( 3 ).ufl_element() )
tau_profile = Expression( 'C / (2.0 * pow((1.0 + C * C) * (x[0]*x[0] + x[1]*x[1]), 3.0/2.0)) ', C=C, element=fsp.Q.sub( 3 ).ufl_element() )

bc_z = DirichletBC( fsp.Q.sub( 0 ), z_profile, rmsh.boundary )
# bc_mu = DirichletBC( fsp.Q.sub( 2 ), mu_profile, rmsh.boundary )
# bc_nu = DirichletBC( fsp.Q.sub( 3 ), nu_profile, rmsh.boundary )
bc_tau = DirichletBC( fsp.Q.sub( 4 ), tau_profile, rmsh.boundary )

# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z]
# bcs = [bc_z, bc_tau]

# Define variational problem

F_z = (kappa * (geo.g_c( fsp.omega )[i, j] * fsp.nu[j] * (fsp.nu_z.dx( i )) - 2.0 * fsp.mu * ((fsp.mu ** 2) - geo.K( fsp.omega )) * fsp.nu_z) + fsp.sigma * fsp.mu * fsp.nu_z) * geo.sqrt_detg(
    fsp.omega ) * dx \
      - ( \
                  + (kappa * (rmsh.n_circle( fsp.omega ))[i] * fsp.nu_z * fsp.nu[i]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
                  + (kappa * (rmsh.n_circle( fsp.omega ))[i] * fsp.nu_z * fsp.nu[i]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
      )

F_omega = (- fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] - fsp.omega[i] * fsp.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * dx \
          + ((rmsh.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
          + ((rmsh.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R

# F_mu =   (   fsp.omega[j] * geo.g_c(fsp.omega)[i, j] * ((fsp.nu_mu / geo.sqrt_detg(fsp.omega_output)).dx(i))  + 2 * fsp.mu * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * dx \
#          - ((rmsh.n_circle( fsp.omega ))[i] * fsp.omega[i] / geo.sqrt_detg(fsp.omega) * fsp.nu_mu) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
#          - ((rmsh.n_circle( fsp.omega ))[i] * fsp.omega[i] / geo.sqrt_detg(fsp.omega) * fsp.nu_mu) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * dx

F_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * dx \
       - ((rmsh.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
       - ((rmsh.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * rmsh.ds_R

F_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * dx \
        - ((rmsh.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
        - ((rmsh.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R

F_N = alpha / rmsh.r_mesh * ( \
            + (((rmsh.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_r) * ((rmsh.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_circle( fsp.omega,
                                                                                                                                                                                     rmsh.c_r ) * (
                    1.0 / rmsh.r) * rmsh.ds_r \
            + (((rmsh.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_R) * ((rmsh.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_circle( fsp.omega,
                                                                                                                                                                                     rmsh.c_R ) * (
                    1.0 / rmsh.R) * rmsh.ds_R \
 \
            + (fsp.mu - geo.H( fsp.omega )) * fsp.nu_mu * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
            + (fsp.mu - geo.H( fsp.omega )) * fsp.nu_mu * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R \
 \
            + (fsp.nu[i] - (fsp.mu.dx( i ))) * fsp.nu_nu[i] * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
            + (fsp.nu[i] - (fsp.mu.dx( i ))) * fsp.nu_nu[i] * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R \
 \
            + (fsp.tau - geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu, fsp.omega )[j, i]) * fsp.nu_tau * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
            + (fsp.tau - geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu, fsp.omega )[j, i]) * fsp.nu_tau * rmsh.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R \
 \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu + F_nu + F_tau) + F_N
