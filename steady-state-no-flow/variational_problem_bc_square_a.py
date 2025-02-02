from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
#bending rigidity
kappa = 1.0
#density
rho = 1.0
#Nitche's parameter
alpha = 1e1

class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
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

class TauExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

class omega_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.5
    def value_shape(self):
        return (1,)

class omega_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_circle = interpolate( omega_circle_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_square = interpolate( omega_square_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
fsp.z_0.interpolate( ManifoldExpression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_0.interpolate( OmegaExpression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_0.interpolate( MuExpression( element=fsp.Q_mu.ufl_element() ) )
fsp.nu_0.interpolate( NuExpression( element=fsp.Q_nu.ufl_element() ) )

fsp.tau_0.interpolate( TauExpression( element=fsp.Q_tau.ufl_element() ) )


# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0, fsp.nu_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_circle = DirichletBC( fsp.Q.sub( 0 ), Expression( '0.0', element=fsp.Q.sub( 0 ).ufl_element() ), rmsh.boundary_circle )
bc_z_square = DirichletBC( fsp.Q.sub( 0 ), Expression( '0.0', element=fsp.Q.sub( 0 ).ufl_element() ), rmsh.boundary_square )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_circle, bc_z_square]

# Define variational problem

F_z = ( kappa * ( geo.g_comega)[i, j] * (geo.H(omega).dx(j)) * (bgeo.nu_z.dx(i)) - 2.0 * geo.H(omega) * ( (geo.H(omega))**2 - K(omega) ) * bgeo.nu_z ) + sigma * geo.H(omega) * bgeo.nu_z ) * sqrt_detg(omega) * dx \
    - ( \
        ( kappa * (bgeo.n_lr(omega))[i] * bgeo.nu_z * (geo.H(omega).dx(i)) ) * bgeo.sqrt_deth_lr(omega) * (ds_l + ds_r) \
        + ( kappa * (bgeo.n_tb(omega))[i] * bgeo.nu_z * (geo.H(omega).dx(i)) ) * bgeo.sqrt_deth_tb(omega) * (ds_t + ds_b) \
        + ( kappa * (bgeo.n_circle(omega))[i] * bgeo.nu_z * (geo.H(omega).dx(i)) ) * bgeo.sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
      )

F_omega = ( - z * Nabla_v(bgeo.nu_omega, omega)[i, i] - omega[i] * bgeo.nu_omega[i] ) *  sqrt_detg(omega) * dx \
          + ( (bgeo.n_lr(omega))[i] * g(omega)[i, j] * z * bgeo.nu_omega[j] ) * bgeo.sqrt_deth_lr(omega) * (ds_l + ds_r) \
          + ( (bgeo.n_tb(omega))[i] * g(omega)[i, j] * z * bgeo.nu_omega[j] ) * bgeo.sqrt_deth_tb(omega) * (ds_t + ds_b) \
          + ( (bgeo.n_circle(omega))[i] * g(omega)[i, j] * z * bgeo.nu_omega[j] ) * bgeo.sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle

F_N = alpha / rmsh.r_mesh * ( \
              + ( ( (bgeo.n_lr(omega))[i] * omega[i] - omega_square ) * ((bgeo.n_lr(omega))[k] * g( omega )[k, l] * bgeo.nu_omega[l]) ) * bgeo.sqrt_deth_lr( omega ) * ( ds_l + ds_r) \
              + ( ( (bgeo.n_tb(omega))[i] * omega[i] - omega_square ) * ((bgeo.n_tb(omega))[k] * g( omega )[k, l] * bgeo.nu_omega[l]) ) * bgeo.sqrt_deth_tb( omega ) * ( ds_t + ds_b) \
              + ( ( (bgeo.n_circle(omega))[i] * omega[i] - omega_circle ) * ((bgeo.n_circle(omega))[k] * g( omega )[k, l] * bgeo.nu_omega[l]) ) * bgeo.sqrt_deth_circle(omega, c_r) * (1.0 / r) * ds_circle \
      )


# total functional for the mixed problem
F = ( F_z + F_omega ) + F_N