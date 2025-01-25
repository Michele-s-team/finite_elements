from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import ufl as ufl

import geometry as geo
import function_spaces as fsp
import read_mesh_ring as rmsh


i, j, k, l = ufl.indices(4)


# CHANGE PARAMETERS HERE
# bending rigidity
kappa = 1.0
# density
rho = 1.0
# viscosity
eta = 1.0
C = 0.1
#values of z at the boundaries
'''
if you compare with the solution from check-with-analytical-solution-bc-ring.nb:
    - z_r(R)_const_{here} <-> zRmin(max)_{check-with-analytical-solution-bc-ring.nb}
    - zp_r(R)_const_{here} <-> zpRmin(max)_{check-with-analytical-solution-bc-ring.nb}
'''
z_r_const = 1.0/10.0
z_R_const = 4.0/5.0
zp_r_const = -3.0/10.0
zp_R_const = 6.0/5.0
omega_r_const = - (rmsh.r) * zp_r_const / sqrt( (rmsh.r)**2  * (1.0 + zp_r_const**2))
omega_R_const = (rmsh.R) * zp_R_const / sqrt( (rmsh.R)**2  * (1.0 + zp_R_const**2))
# Nitche's parameter
alpha = 1e1


class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        # values[0] = (2.0 + C**2) * kappa / (2.0 * (1.0 + C**2) * (x[0]**2 + x[1]**2))
        values[0] =  cos(2.0*(np.pi)*geo.my_norm(x))

    def value_shape(self):
        return (1,)


class ManifoldExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class OmegaExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class MuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class NuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


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


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
fsp.z_0.interpolate( ManifoldExpression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_0.interpolate( OmegaExpression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_0.interpolate( MuExpression( element=fsp.Q_mu.ufl_element() ) )
fsp.nu_0.interpolate( NuExpression( element=fsp.Q_nu.ufl_element() ) )

# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [z_0, omega_0, mu_0, nu_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_r = DirichletBC( fsp.Q.sub( 0 ), Expression( 'z_r', z_r=z_r_const, element=fsp.Q.sub( 0 ).ufl_element() ), rmsh.boundary_r )
bc_z_R = DirichletBC( fsp.Q.sub( 0 ), Expression( 'z_R', z_R=z_R_const, element=fsp.Q.sub( 0 ).ufl_element() ), rmsh.boundary_R )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_r, bc_z_R]

# Define variational problem

F_z = (kappa * (geo.g_c( fsp.omega )[i, j] * fsp.nu[j] * (fsp.nu_z.dx( i )) - 2.0 * fsp.mu * ((fsp.mu ** 2) - geo.K( fsp.omega )) * fsp.nu_z) + fsp.sigma * fsp.mu * fsp.nu_z) * geo.sqrt_detg( fsp.omega ) * dx \
      - ( \
                  + (kappa * (geo.n_circle( fsp.omega ))[i] * fsp.nu_z * fsp.nu[i]) * geo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * ds_r \
                  + (kappa * (geo.n_circle( fsp.omega ))[i] * fsp.nu_z * fsp.nu[i]) * geo.sqrt_deth_circle( fsp.omega, c_R ) * (1.0 / R) * ds_R
      )

F_omega = (- z * Nabla_v( nu_omega, fsp.omega )[i, i] - fsp.omega[i] * nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * dx \
          + ((geo.n_circle( fsp.omega ))[i] * g( fsp.omega )[i, j] * z * nu_omega[j]) * geo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * ds_r \
          + ((geo.n_circle( fsp.omega ))[i] * g( fsp.omega )[i, j] * z * nu_omega[j]) * geo.sqrt_deth_circle( fsp.omega, c_R ) * (1.0 / R) * ds_R

F_mu = ((H( fsp.omega ) - fsp.mu) * nu_mu) * geo.sqrt_detg( fsp.omega ) * dx

F_nu = (fsp.nu[i] * nu_nu[i] + fsp.mu * Nabla_v( nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * dx \
       - ((geo.n_circle( fsp.omega ))[i] * g( fsp.omega )[i, j] * fsp.mu * nu_nu[j]) * geo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * ds_r \
       - ((geo.n_circle( fsp.omega ))[i] * g( fsp.omega )[i, j] * fsp.mu * nu_nu[j]) * geo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / R) * ds_R

F_N = alpha / r_mesh * ( \
            + (((geo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_r) * ((geo.n_circle( fsp.omega ))[k] * g( fsp.omega )[k, l] * nu_omega[l])) * geo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * ds_r \
            + (((geo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_R) * ((geo.n_circle( fsp.omega ))[k] * g( fsp.omega )[k, l] * nu_omega[l])) * geo.sqrt_deth_circle( fsp.omega, c_R ) * (1.0 / R) * ds_R \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu + F_nu) + F_N
