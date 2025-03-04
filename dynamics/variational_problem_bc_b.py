from fenics import *
from mshr import *
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



v_bar_l_const = (float)( rarg.args.v )
z_square_const = 0.0
omega_n_square_const = 0.0
omega_n_circle_const = 0.1
omega_r_circle_const = 0.1


alpha = 1e4

class v_bar_l_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = v_bar_l_const
        values[1] = 0

    def value_shape(self):
        return (2,)

class v_bar_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

class w_bar_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (2,)

class phi_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (2,)


class z_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = z_square_const

    def value_shape(self):
        return (2,)


class omega_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_r_circle_const * (x[0] - rmsh.c_r[0]) / geo.my_norm( x - rmsh.c_r )
        values[1] = omega_r_circle_const * (x[1] - rmsh.c_r[1]) / geo.my_norm( x - rmsh.c_r )

    def value_shape(self):
        return (2,)

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


class omega_n_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_n_square_const

    def value_shape(self):
        return (1,)


# profiles for the normal derivative
class omega_n_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_n_circle_const

    def value_shape(self):
        return (1,)


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_n_circle = interpolate( omega_n_circle_Expression( element=fsp.Q_z_n.ufl_element() ), fsp.Q_z_n )
omega_n_square = interpolate( omega_n_square_Expression( element=fsp.Q_z_n.ufl_element() ), fsp.Q_z_n )

# boundary conditions (BCs)
# BCs for v_bar
v_bar_l = interpolate( v_bar_l_Expression( element=fsp.Q_v_bar.ufl_element() ), fsp.Q_v_bar )
bc_v_bar_l = DirichletBC( fsp.Q.sub( 0 ), v_bar_l, rmsh.boundary_l )

v_bar_circle = interpolate( v_bar_circle_Expression( element=fsp.Q_v_bar.ufl_element() ), fsp.Q_v_bar )
bc_v_bar_circle = DirichletBC( fsp.Q.sub( 0 ), v_bar_circle, rmsh.boundary_circle )

# BCs for w_bar
w_bar_square = interpolate( w_bar_square_Expression( element=fsp.Q_w_bar.ufl_element() ), fsp.Q_w_bar )
bc_w_bar_square = DirichletBC( fsp.Q.sub( 1 ), w_bar_square, rmsh.boundary_square)

# BC for phi
phi_r = interpolate( phi_r_Expression( element=fsp.Q_phi.ufl_element() ), fsp.Q_phi)
bc_phi_r = DirichletBC( fsp.Q.sub( 2 ), phi_r, rmsh.boundary_r )

z_square = interpolate( z_square_Expression( element=fsp.Q_z_n.ufl_element() ), fsp.Q_z_n )
bc_z_square = DirichletBC( fsp.Q.sub( 5 ), z_square, rmsh.boundary_square )

omega_circle = interpolate( omega_circle_Expression( element=fsp.Q_omega_n.ufl_element() ), fsp.Q_omega_n )
bc_omega_circle = DirichletBC( fsp.Q.sub( 6 ), omega_circle, rmsh.boundary_circle )
# CHANGE PARAMETERS HERE


# all BCs
bcs = [bc_v_bar_l, bc_v_bar_circle, bc_w_bar_square, bc_phi_r, bc_z_square, bc_omega_circle]

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
                      ((fsp.W ** 2) * (bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                      + ((fsp.W ** 2) * (bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
                      + ((fsp.W ** 2) * (bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          - dt * ( \
                      (fsp.sigma_n_32 * (bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                      + (fsp.sigma_n_32 * (bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
                      + (fsp.sigma_n_32 * (bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          - dt * 2.0 * eta * ( \
                      (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_lr( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_l \
                      # natural BC imposed here
                      + (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, 1] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_lr( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[1]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_r \
                      + (geo.d_c( fsp.V, fsp.W, fsp.omega_n_12 )[i, j] * geo.g( fsp.omega_n_12 )[i, k] * (bgeo.n_tb( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
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
                      (fsp.W * fsp.nu_w_bar * (bgeo.n_lr( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                      + (fsp.W * fsp.nu_w_bar * (bgeo.n_tb( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
                      + (fsp.W * fsp.nu_w_bar * (bgeo.n_circle( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (
                              1.0 / rmsh.r) * rmsh.ds_circle
          ) \
          + dt * 2.0 * kappa * ( \
                      (fsp.nu_w_bar * (bgeo.n_lr( fsp.omega_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                      + (fsp.nu_w_bar * (bgeo.n_tb( fsp.omega_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
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
                        ((bgeo.n_lr( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                        + ((bgeo.n_tb( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
                        + ((bgeo.n_circle( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
            )

F_mu_n = ((geo.H( fsp.omega_n_12 ) - fsp.mu_n_12) * fsp.nu_mu_n_12) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx


F_N = alpha / rmsh.r_mesh * ( \
 \
            (fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_tb( fsp.omega_n_12 ))[j] * (bgeo.n_tb( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
 \
            + (((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - omega_n_square) * (bgeo.n_lr( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
            + (((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - omega_n_square) * (bgeo.n_tb( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
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
