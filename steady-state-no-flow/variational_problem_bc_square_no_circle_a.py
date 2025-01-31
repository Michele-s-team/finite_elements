from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import read_mesh_square_no_circle as rmsh

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
#bending rigidity
kappa = 1.0
sigma0 = 1.0
C = 0.1
D = -0.1
#Nitche's parameter
alpha = 1e1

class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = sigma0
    def value_shape(self):
        return (1,)

class z0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0000300371897 - 0.00524403086 * x[1] + 0.844347452 * x[1]**2 - 0.738886103 * x[1]**3
    def value_shape(self):
        return (1,)

class omega0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.002597872 + 1.560845 * x[1] - 1.829453 * x[1]**2 - 0.2990343 * x[1]**3
    def value_shape(self):
        return (2,)

class omega_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = D * x[1]/rmsh.h
    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_square = interpolate( omega_square_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ))
fsp.omega_0.interpolate( omega0_Expression( element=fsp.Q_omega.ufl_element() ))
fsp.z_0.interpolate( z0_Expression( element=fsp.Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0, fsp.nu_0, fsp.tau_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_square = DirichletBC( fsp.Q.sub( 0 ), Expression( 'C * x[1]/h', element=fsp.Q.sub( 0 ).ufl_element(), C=C, h=rmsh.h), rmsh.boundary_square )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_square]

# Define variational problem

F_z = ( kappa * ( geo.g_c(fsp.omega)[i, j] * (geo.H(fsp.omega).dx(j)) * (fsp.nu_z.dx(i)) - 2.0 * geo.H(fsp.omega) * ( (geo.H(fsp.omega))**2 - geo.K(fsp.omega) ) * fsp.nu_z ) + fsp.sigma * geo.H(fsp.omega) * fsp.nu_z ) * geo.sqrt_detg(fsp.omega) * dx \
    - ( \
        ( kappa * (rmsh.n_lr(fsp.omega))[i] * fsp.nu_z * (geo.H(fsp.omega).dx(i)) ) * rmsh.sqrt_deth_lr(fsp.omega) * (rmsh.ds_l + rmsh.ds_r) \
        + ( kappa * (rmsh.n_tb(fsp.omega))[i] * fsp.nu_z * (geo.H(fsp.omega).dx(i)) ) * rmsh.sqrt_deth_tb(fsp.omega) * (rmsh.ds_t + rmsh.ds_b) \
      )

F_omega = ( - fsp.z * geo.Nabla_v(fsp.nu_omega, fsp.omega)[i, i] - fsp.omega[i] * fsp.nu_omega[i] ) *  geo.sqrt_detg(fsp.omega) * rmsh.dx \
          + ( (rmsh.n_lr(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.z * fsp.nu_omega[j] ) * rmsh.sqrt_deth_lr(fsp.omega) * (rmsh.ds_l + rmsh.ds_r) \
          + ( (rmsh.n_tb(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.z * fsp.nu_omega[j] ) * rmsh.sqrt_deth_tb(fsp.omega) * (rmsh.ds_t + rmsh.ds_b) \


F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((rmsh.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * rmsh.sqrt_deth_lr( fsp.omega) * (rmsh.ds_l + rmsh.ds_r)  \
       - ((rmsh.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * rmsh.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b)

F_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
        - ((rmsh.n_lr( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * rmsh.sqrt_deth_lr( fsp.omega  ) * (rmsh.ds_l + rmsh.ds_r) \
        - ((rmsh.n_tb( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * rmsh.sqrt_deth_tb( fsp.omega) * (rmsh.ds_t + rmsh.ds_b)

F_N = alpha / rmsh.r_mesh * ( \
              + ( ( (rmsh.n_lr(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((rmsh.n_lr(fsp.omega))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * rmsh.sqrt_deth_lr( fsp.omega ) * ( rmsh.ds_l + rmsh.ds_r) \
              + ( ( (rmsh.n_tb(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((rmsh.n_tb(fsp.omega))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * rmsh.sqrt_deth_tb( fsp.omega ) * ( rmsh.ds_t + rmsh.ds_b) \
      )


# total functional for the mixed problem
F = ( F_z + F_omega ) + F_N