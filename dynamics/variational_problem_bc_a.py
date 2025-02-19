from fenics import *
from mshr import *
import numpy as np
import ufl as ufl


import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh as rmsh
import runtime_arguments as rarg

i, j, k, l = ufl.indices( 4 )


# CHANGE PARAMETERS HERE
# time step size
T = (float)( rarg.args.T )
N = (int)( rarg.args.N )
dt = T / N
# time step size
# bending rigidity
kappa = (float)( rarg.args.k )
# density
rho = (float)( rarg.args.r )
# viscosity
eta = (float)( rarg.args.e )
# inflow velocity
v_l = (float)( rarg.args.v )
# value of w_bar at the boundary
boundary_profile_w_bar = 0.0
# value of phi ar r boundary
r_profile_phi = 0.0
#value of z at boundary
boundary_profile_z = 0.0
# Nitche's parameter
alpha = 1e4


class TangentVelocityExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class NormalVelocityExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0

    def value_shape(self):
        return (1,)


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


class grad_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


# profiles for the normal derivative
class grad_circle_Expression( UserExpression ):
    def eval(self, values, x):
        a = 0.1
        values[0] = a

    def value_shape(self):
        return (1,)


# CHANGE PARAMETERS HERE

# Define expressions used in variational forms
# Deltat = Constant( dt )
# kappa = Constant( kappa )
# rho = Constant( rho )

# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
grad_circle = interpolate( grad_circle_Expression( element=fsp.Q_z_n.ufl_element() ), fsp.Q_z_n )
grad_square = interpolate( grad_square_Expression( element=fsp.Q_z_n.ufl_element() ), fsp.Q_z_n )

# CHANGE PARAMETERS HERE
l_profile_v_bar = Expression( ('v_l', '0'), v_l=v_l, element=fsp.Q_v_n.ufl_element() )
# CHANGE PARAMETERS HERE

# boundary conditions (BCs)
# BCs for v_bar
bc_v_bar_l = DirichletBC( fsp.Q.sub( 0 ), l_profile_v_bar, rmsh.boundary_l )

# BCs for w_bar
bc_w_bar_lr = DirichletBC( fsp.Q.sub( 1 ), Constant( boundary_profile_w_bar ), rmsh.boundary_lr )
bc_w_bar_tb = DirichletBC( fsp.Q.sub( 1 ), Constant( boundary_profile_w_bar ), rmsh.boundary_tb )
bc_w_bar_circle = DirichletBC( fsp.Q.sub( 1 ), Constant( boundary_profile_w_bar ), rmsh.boundary_circle )

# BC for phi
bc_phi = DirichletBC( fsp.Q.sub( 2 ), Constant( r_profile_phi ), rmsh.boundary_r )

# CHANGE PARAMETERS HERE
# BCs for z^{n-1/2}
bc_z_circle = DirichletBC( fsp.Q.sub( 5 ), Expression( 'boundary_profile_z', element=fsp.Q.sub( 5 ).ufl_element(), boundary_profile_z=boundary_profile_z ), rmsh.boundary_circle )
bc_z_square = DirichletBC( fsp.Q.sub( 5 ), Expression( 'boundary_profile_z', element=fsp.Q.sub( 5 ).ufl_element(), boundary_profile_z=boundary_profile_z ), rmsh.boundary_square )
# CHANGE PARAMETERS HERE


# all BCs
bcs = [bc_v_bar_l, bc_w_bar_lr, bc_w_bar_tb, bc_w_bar_circle, bc_phi, bc_z_circle, bc_z_square]

'''
Define variational problem : F_vbar, F_wbar .... F_z_n_12 are related to the PDEs for v_bar, ..., z^{n-1/2} respectively . F_N enforces the BCs with Nitsche's method. 
To be safe, I explicitly wrote the each term on each part of the boundary with its own normal vector and pull back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle, 
the surface elements are ds_l + ds_r, and the normal is n_lr(omega) and the pull back of the metric is sqrt_deth_lr(omega): this avoids odd interpolations at the corners of the rectangle edges. 
'''

F_v_bar = ( \
                      rho * (( \
                                         (fsp.v_bar[i] - fsp.v_n_1[i]) \
                                         + dt * ((3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * geo.Nabla_v( fsp.V, fsp.omega_n_12 )[i, j] \
                                                     - 2.0 * fsp.V[j] * fsp.W * geo.g_c( fsp.omega_n_12 )[i, k] * geo.b( fsp.omega_n_12 )[k, j]) \
 \
                                 ) * fsp.nu_v_bar[i] \
                             + dt * 1.0 / 2.0 * (fsp.W ** 2) * geo.g_c( fsp.omega_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.omega_n_12 )[i, j] \
                             ) \
                      + dt * (fsp.sigma_n_32 * geo.g_c( fsp.omega_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.omega_n_12 )[i, j] \
                                  + 2.0 * eta * geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.omega_n_12 )[j, i])
          ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
          - dt * rho / 2.0 * ( \
                      ((fsp.W ** 2) * (bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                      + ((fsp.W ** 2) * (bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                      + ((fsp.W ** 2) * (bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          - dt * ( \
                      (fsp.sigma_n_32 * (bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                      + (fsp.sigma_n_32 * (bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                      + (fsp.sigma_n_32 * (bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          - dt * 2.0 * eta * ( \
                      (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_lr( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_l \
                      # natural BC imposed here
                      + (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, 1] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_lr( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[1]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_r \
                      + (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_tb( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                      + (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_circle( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          )

F_w_bar = ( \
                      rho * ((fsp.w_bar - fsp.w_n_1) + dt * fsp.V[i] * fsp.V[k] * geo.b( fsp.omega_n_12 )[k, i]) * fsp.nu_w_bar \
                      - dt * rho * fsp.W * geo.Nabla_v( geo.vector_times_scalar( 3.0 / 2.0 * fsp.v_n_1 - 1.0 / 2.0 * fsp.v_n_2, fsp.nu_w_bar ), fsp.omega_n_12 )[i, i] \
                      + dt * 2.0 * kappa * ( \
                                  - geo.g_c( fsp.omega_n_12 )[i, j] * ((fsp.mu_n_12).dx( j )) * (fsp.nu_w_bar.dx( i )) \
                                  + 2.0 * fsp.mu_n_12 * (((fsp.mu_n_12) ** 2) - geo.K( fsp.omega_n_12 )) * fsp.nu_w_bar \
                          ) \
                      - dt * ( \
                                  2.0 * fsp.sigma_n_32 * fsp.mu_n_12 \
                                  + 2.0 * eta * (geo.g_c( fsp.omega_n_12 )[i, k] * geo.Nabla_v( fsp.V, fsp.omega_n_12 )[j, k] *
                                                 (geo.b( fsp.omega_n_12 ))[i, j] - 2.0 * fsp.W * (
                                                         2.0 * ((fsp.mu_n_12) ** 2) - geo.K( fsp.omega_n_12 )))
                      ) * fsp.nu_w_bar
          ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
          + dt * rho * ( \
                      (fsp.W * fsp.nu_w_bar * (bgeo.n_lr( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                      + (fsp.W * fsp.nu_w_bar * (bgeo.n_tb( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                      + (fsp.W * fsp.nu_w_bar * (bgeo.n_circle( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (
                              1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          + dt * 2.0 * kappa * ( \
                      (fsp.nu_w_bar * (bgeo.n_lr( fsp.omega_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                      + (fsp.nu_w_bar * (bgeo.n_tb( fsp.omega_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                      + (fsp.nu_w_bar * (bgeo.n_circle( fsp.omega_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          )

F_phi = ( \
                    dt * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.phi.dx( i )) * (fsp.nu_phi.dx( j )) \
                    + rho * (geo.Nabla_v( fsp.v_bar, fsp.omega_n_12 )[i, i] - 2.0 * fsp.mu_n_12 * fsp.w_bar) * fsp.nu_phi \
            ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
    # natural BC implemented here
- dt * ((bgeo.n_lr( fsp.omega_n_12 ))[i] * (fsp.phi.dx( i )) * fsp.nu_phi) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_r

F_v_n = ((rho * (fsp.v_n[i] - fsp.v_bar[i]) + dt * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.phi.dx( j ))) * fsp.nu_v_n[i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_w_n = ((fsp.w_n - fsp.w_bar) * fsp.nu_w_n) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_z_n = ( \
                    ( \
                                (fsp.z_n_12 - fsp.z_n_32) \
                                - dt * fsp.w_n_1 * ((geo.normal( fsp.omega_n_12 ))[2] - ((geo.normal( fsp.omega_n_12 ))[0] * fsp.omega_n_12[0] + (geo.normal( fsp.omega_n_12 ))[1] * fsp.omega_n_12[1])) \
                        ) * fsp.nu_z_n_12 \
            ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_omega_n = (fsp.z_n_12 * geo.Nabla_v( fsp.nu_omega_n_12, fsp.omega_n_12 )[i, i] + fsp.omega_n_12[i] * fsp.nu_omega_n_12[i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
            - ( \
                        ((bgeo.n_lr( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                        + ((bgeo.n_tb( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                        + ((bgeo.n_circle( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
            )

F_mu_n = ((geo.H( fsp.omega_n_12 ) - fsp.mu_n_12) * fsp.nu_mu_n_12) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx


F_N = alpha / rmsh.r_mesh * ( \
 \
            (fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_tb( fsp.omega_n_12 ))[j] * (bgeo.n_tb( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
            + (fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_circle( fsp.omega_n_12 ))[j] * (bgeo.n_circle( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[k]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle \
 \
            + (((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_square) * (bgeo.n_lr( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
            + (((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_square) * (bgeo.n_tb( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
            + (((bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_circle) * (bgeo.n_circle( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (
                    1.0 / rmsh.r) * rmsh.ds_circle \
    )

# total functional for the mixed problem
F = (F_v_bar + F_w_bar + F_phi + F_v_n + F_w_n + F_z_n + F_omega_n + F_mu_n) + F_N

#post-processing variational functional
F_pp_nu = (fsp.nu_n_12[i] * fsp.nu_nu[i] + fsp.mu_n_12 * geo.Nabla_v( fsp.nu_nu, fsp.omega_n_12 )[i, i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
       - ((bgeo.n_tb( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
       - ((bgeo.n_circle( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r

F_pp_tau = (fsp.nu_n_12[i] * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau_n_12 * fsp.nu_tau) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
           - ((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
           - ((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
           - ((bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r

F_pp_d = ((geo.d(fsp.V, fsp.W, fsp.omega_n_12)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx
