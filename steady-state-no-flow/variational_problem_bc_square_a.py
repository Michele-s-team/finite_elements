from fenics import *
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
# bending rigidity
kappa = 1.0
# Nitche's parameter
alpha = 1e2

z_circle_const = 0.0
z_square_const = 0.1
omega_circle_const = 0.5
omega_square_const = 0.0


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


class MuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

class NuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

class TauExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class omega_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_circle_const

    def value_shape(self):
        return (1,)


class omega_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_square_const

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
fsp.assigner.assign( fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0] )

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_circle = DirichletBC( fsp.Q.sub( 0 ), Expression( 'z_circle_const', element=fsp.Q.sub( 0 ).ufl_element(), z_circle_const=z_circle_const ), rmsh.boundary_circle )
bc_z_square = DirichletBC( fsp.Q.sub( 0 ), Expression( 'z_square_const', element=fsp.Q.sub( 0 ).ufl_element(), z_square_const=z_square_const ), rmsh.boundary_square )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_circle, bc_z_square]

# Define variational problem

F_z = (kappa * (geo.g_c( fsp.omega )[i, j] * (fsp.mu.dx(j)) * (fsp.nu_z.dx( i )) - 2.0 * fsp.mu * (
        (fsp.mu) ** 2 - geo.K( fsp.omega )) * fsp.nu_z) + fsp.sigma * fsp.mu * fsp.nu_z) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - ( \
                  (kappa * (bgeo.n_lr( fsp.omega ))[i] * fsp.nu_z * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
                  + (kappa * (bgeo.n_tb( fsp.omega ))[i] * fsp.nu_z * (fsp.mu.dx(i))) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
                  + (kappa * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_z * (fsp.mu.dx(i))) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
      )

F_omega = (- fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] - fsp.omega[i] * fsp.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          + ((bgeo.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
          + ((bgeo.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
          + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_N = alpha / rmsh.r_mesh * ( \
            + (((bgeo.n_lr( fsp.omega ))[i] * fsp.omega[i] - omega_square) * ((bgeo.n_lr( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
            + (((bgeo.n_tb( fsp.omega ))[i] * fsp.omega[i] - omega_square) * ((bgeo.n_tb( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
            + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_circle) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu ) + F_N

# post-processing variational functionals
F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
       - ((bgeo.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle

F_pp_tau = ((fsp.mu.dx( i )) * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
           - ((bgeo.n_lr( fsp.omega ))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
           - ((bgeo.n_tb( fsp.omega ))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_circle( fsp.omega , rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle
