from __future__ import print_function
from fenics import *
from mshr import *
from fenics import *
from mshr import *
from geometry import *

# CHANGE PARAMETERS HERE
#bending rigidity
kappa = 1.0
#density
rho = 1.0
#viscosity
eta = 1.0
#Nitche's parameter
alpha = 1e1

class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1.0
    def value_shape(self):
        return (1,)

class ManifoldExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
    def value_shape(self):
        return (1,)

class OmegaExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.5
    def value_shape(self):
        return (1,)

class omega_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_r = interpolate( omega_r_Expression( element=Q_z.ufl_element() ), Q_z )
omega_R = interpolate( omega_R_Expression( element=Q_z.ufl_element() ), Q_z )

sigma.interpolate( SurfaceTensionExpression( element=Q_sigma.ufl_element() ))
omega_0.interpolate( OmegaExpression( element=Q_omega.ufl_element() ))
z_0.interpolate( ManifoldExpression( element=Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [omega_0, z_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_r = DirichletBC( Q.sub( 1 ), Expression( '0.1', element=Q.sub( 1 ).ufl_element() ), boundary_r )
bc_z_R = DirichletBC( Q.sub( 1 ), Expression( '0.0', element=Q.sub( 1 ).ufl_element() ), boundary_R )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_r, bc_z_R]

# Define variational problem

F_z = ( kappa * ( g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 2.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) + sigma * H(omega) * nu_z ) * sqrt_detg(omega) * dx \
    - ( \
        + ( kappa * (n_circle(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
        + ( kappa * (n_circle(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_circle( omega, c_R ) * (1.0 / R) * ds_R
      )

F_omega = ( - z * Nabla_v(nu_omega, omega)[i, i] - omega[i] * nu_omega[i] ) *  sqrt_detg(omega) * dx \
          + ( (n_circle(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
          + ( (n_circle(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_circle( omega, c_R ) * (1.0 / R) * ds_R

F_N = alpha / r_mesh * ( \
              + ( ( (n_circle(omega))[i] * omega[i] - omega_r ) * ((n_circle(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_circle(omega, c_r) * (1.0 / r) * ds_r \
              + ( ( (n_circle(omega))[i] * omega[i] - omega_R ) * ((n_circle(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_circle(omega, c_R) * (1.0 / R) * ds_R \
      )


# total functional for the mixed problem
F = ( F_z + F_omega ) + F_N