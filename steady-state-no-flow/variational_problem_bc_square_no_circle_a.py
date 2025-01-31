from fenics import *
import ufl as ufl
import numpy as np

import function_spaces as fsp
import geometry as geo
import read_mesh_square_no_circle as rmsh

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
#bending rigidity
kappa = 1.0
#Nitche's parameter
alpha = 1e1


class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = (16 * (-1 + 4 * x[0]**6 + 6 * x[1]**2 + 6 * x[1]**4 + 4 * x[1]**6 + \
                  6 * x[0]**4 * (1 + 2 * x[1]**2) +\
                  6 * x[0]**2 * (1 + 2 * (x[1]**2 + x[1]**4))) * kappa) / \
           ((1 + 2 * x[0]**2 + 2 * x[1]**2) * (1 + 4 * x[0]**2 + 4 * x[1]**2)**3)

    def value_shape(self):
        return (1,)


class z_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = x[0]**2 + x[1]**2

    def value_shape(self):
        return (1,)


class omega_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 2 * x[0]
        values[1] = 2 * x[1]

    def value_shape(self):
        return (2,)


class mu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (4 + 8 * x[0]**2 + 8 * x[1]**2) / (2 * (1 + 4 * x[0]**2 + 4 * x[1]**2)**(3.0/2.0))


    def value_shape(self):
        return (1,)


class nu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = - (16 * x[0] * (1 + x[0]**2 + x[1]**2)) / ((1 + 4 * x[0]**2 + 4 * x[1]**2) ** (5.0/2.0))
        values[1] = -((16 * x[1] * (1 + x[0]**2 + x[1]**2)) / (1 + 4 * x[0]**2 + 4 * x[1]**2)**(5.0/2.0))

    def value_shape(self):
        return (2,)


class tau_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (32 * (-1 + 4 * x[0]**4 + 6 * x[1]**2 + 4 * x[1]**4 + x[0]**2 * (6 + 8 * x[1]**2))) / (1 + 4 * x[0]**2 + 4 * x[1]**2)**(9.0/2.0)


    def value_shape(self):
        return (1,)


class omega_l_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -((8 * x[1]**2) / np.sqrt((1 + 4 * x[1]**2) * (5 + 4 * x[1]**2))) + 2 / np.sqrt(1 + 4 / (1 + 4 * x[1]**2))

    def value_shape(self):
        return (1,)


class omega_t_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -((8 * x[0]**2) / np.sqrt((1 + 4 * x[0]**2) * (5 + 4 * x[0]**2))) + 2 / np.sqrt(1 + 4 / (1 + 4 * x[0]**2))

    def value_shape(self):
        return (1,)

class omega_b_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_l = interpolate( omega_l_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_t = interpolate( omega_t_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_b = interpolate( omega_b_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )


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

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0, fsp.nu_0, fsp.tau_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z = DirichletBC( fsp.Q.sub( 0 ), fsp.z_exact, rmsh.boundary )
bc_mu = DirichletBC( fsp.Q.sub( 2 ), fsp.mu_exact, rmsh.boundary )
bc_nu = DirichletBC( fsp.Q.sub( 3 ), fsp.nu_exact, rmsh.boundary )
bc_tau = DirichletBC( fsp.Q.sub( 4 ), fsp.tau_exact, rmsh.boundary )
# CHANGE PARAMETERS HERE

# all BCs
# bcs = [bc_z, bc_mu, bc_nu, bc_tau]
bcs = [bc_z]

# Define variational problem

F_z = ( kappa * ( geo.g_c(fsp.omega)[i, j] * fsp.nu[j] * (fsp.nu_z.dx(i)) - 2.0 * fsp.mu * ( (fsp.mu)**2 - geo.K(fsp.omega) ) * fsp.nu_z ) + fsp.sigma * fsp.mu * fsp.nu_z ) * geo.sqrt_detg(fsp.omega) * rmsh.dx \
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
            + (((rmsh.n_lr(fsp.omega))[i] * fsp.omega[i] - omega_l) * ((rmsh.n_lr( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_lr( fsp.omega ) * rmsh.ds_l \
            + (((rmsh.n_lr(fsp.omega))[i] * fsp.omega[i] - omega_r) * ((rmsh.n_lr( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_lr( fsp.omega ) * rmsh.ds_r \
\
            + (((rmsh.n_tb(fsp.omega))[i] * fsp.omega[i] - omega_t) * ((rmsh.n_tb( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_tb( fsp.omega ) * rmsh.ds_t\
            + (((rmsh.n_tb(fsp.omega))[i] * fsp.omega[i] - omega_b) * ((rmsh.n_tb( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * rmsh.sqrt_deth_tb( fsp.omega ) * rmsh.ds_b \
      )


# total functional for the mixed problem
F = ( F_z + F_omega + F_mu + F_nu + F_tau) + F_N