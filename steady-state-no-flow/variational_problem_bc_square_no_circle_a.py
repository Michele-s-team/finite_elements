from __future__ import print_function
from fenics import *
from mshr import *
from fenics import *
from mshr import *
from geometry import *

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
        values[0] = D * x[1]/h
    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_square = interpolate( omega_square_Expression( element=Q_z.ufl_element() ), Q_z )

sigma.interpolate( SurfaceTensionExpression( element=Q_sigma.ufl_element() ))
omega_0.interpolate( omega0_Expression( element=Q_omega.ufl_element() ))
z_0.interpolate( z0_Expression( element=Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [omega_0, z_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_square = DirichletBC( Q.sub( 1 ), Expression( 'C * x[1]/h', element=Q.sub( 1 ).ufl_element(), C=C, h=h), boundary_square )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_square]

# Define variational problem

F_z = ( kappa * ( g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 2.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) + sigma * H(omega) * nu_z ) * sqrt_detg(omega) * dx \
    - ( \
        ( kappa * (n_lr(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_lr(omega) * (ds_l + ds_r) \
        + ( kappa * (n_tb(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_tb(omega) * (ds_t + ds_b) \
      )

F_omega = ( - z * Nabla_v(nu_omega, omega)[i, i] - omega[i] * nu_omega[i] ) *  sqrt_detg(omega) * dx \
          + ( (n_lr(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_lr(omega) * (ds_l + ds_r) \
          + ( (n_tb(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_tb(omega) * (ds_t + ds_b) \

F_N = alpha / r_mesh * ( \
              + ( ( (n_lr(omega))[i] * omega[i] - omega_square ) * ((n_lr(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_lr( omega ) * ( ds_l + ds_r) \
              + ( ( (n_tb(omega))[i] * omega[i] - omega_square ) * ((n_tb(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_tb( omega ) * ( ds_t + ds_b) \
      )


# total functional for the mixed problem
F = ( F_z + F_omega ) + F_N