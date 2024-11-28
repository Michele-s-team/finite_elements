from __future__ import print_function
from fenics import *
from mshr import *
from fenics import *
from mshr import *
from geometry import *


# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_v, nu_w, nu_sigma, nu_omega, nu_z = TestFunctions( Q )
# fields at the preceeding steps
# v_n_1 = Function(Q_v_n)
# v_n_2 = Function(Q_v_n)
# w_n_1 = Function(Q_w_n)
# sigma_n_12 = Function( Q_phi )
# sigma_n_32 = Function( Q_phi )
# z_n_32 = Function( Q_z_n )

# v_n_0, ...., z_n_0 are used to store the initial conditions
# sigma_n_12_0 = Function( Q_phi )
v_0 = Function( Q_v )
w_0 = Function( Q_w )
sigma_0 = Function( Q_sigma )
z_0 = Function( Q_z )
omega_0 = Function( Q_omega )

v, w, sigma, omega, z = split( psi )


# CHANGE PARAMETERS HERE
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


# profiles for the normal derivative
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
omega_circle = interpolate( omega_circle_Expression( element=Q_z.ufl_element() ), Q_z )
omega_square = interpolate( omega_square_Expression( element=Q_z.ufl_element() ), Q_z )

# assigner = FunctionAssigner(Q, [Q_v_bar, Q_w_bar, Q_phi, Q_v_n, Q_w_n, Q_omega_n, Q_z_n])
# assigner.assign(psi, [v_bar_0, w_bar_0, phi_0, v_n_0, w_n_0, omega_n_0, z_n_0])

# CHANGE PARAMETERS HERE
l_profile_v = Expression( ('v_l', '0'), v_l=v_l, element = Q_v.ufl_element() )
# CHANGE PARAMETERS HERE

# boundary conditions (BCs)
# BCs for v_bar
bc_v_l = DirichletBC( Q.sub( 0 ), l_profile_v, boundary_l )

# BCs for w_bar
bc_w_lr = DirichletBC( Q.sub( 1 ), Constant( 0 ), boundary_lr )
bc_w_tb = DirichletBC( Q.sub( 1 ), Constant( 0 ), boundary_tb )
bc_w_circle = DirichletBC( Q.sub( 1 ), Constant( 0 ), boundary_circle )

#BC for sigma
bc_sigma = DirichletBC(Q.sub(2), Constant(0), boundary_r)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_circle = DirichletBC( Q.sub( 4 ), Expression( '0.0', element=Q.sub( 4 ).ufl_element() ), boundary_circle )
bc_z_square = DirichletBC( Q.sub( 4 ), Expression( '0.0', element=Q.sub( 4 ).ufl_element(), h=h ), boundary_square )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_v_l, bc_w_lr, bc_w_tb, bc_w_circle, bc_sigma, bc_z_circle, bc_z_square]

#set initial profiles
# v_n_1.interpolate(TangentVelocityExpression(element=Q_v_n.ufl_element()))
# v_n_2.assign(v_n_1)
# w_n_1.interpolate(NormalVelocityExpression(element=Q_w_n.ufl_element()))
# sigma_n_32.interpolate( SurfaceTensionExpression( element=Q_phi.ufl_element() ))
# z_n_32.interpolate( ManifoldExpression( element=Q_z_n.ufl_element() ) )
# omega_n_32.interpolate( OmegaExpression( element=Q_omega_n.ufl_element() ))

# Define variational problem : F_v, F_z are related to the PDEs for v, ..., z respectively . F_N enforces the BCs with Nitsche's method.
# To be safe, I explicitly wrote each term on each part of the boundary with its own normal vector and pull-back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle,
# the surface elements are ds_l + ds_r, and the normal is n_lr(omega), and the pull-back of the metric is sqrt_deth_lr: this avoids odd interpolations at the corners of the rectangle edges.

F_sigma = (Nabla_v( v, omega )[i, i] - 2.0 * H( omega ) * w) * nu_sigma * sqrt_detg( omega ) * dx

F_v = ( \
                    rho * ( \
                          (v[j] * Nabla_v( v, omega )[i, j] - 2.0 * v[j] * w * g_c( omega )[i, k] * b( omega )[k, j]) * nu_v[i] \
                          + 1.0 / 2.0 * (w ** 2) * g_c( omega )[i, j] * Nabla_f( nu_v, omega )[i, j] \
                  ) \
                    + (sigma * g_c( omega )[i, j] * Nabla_f( nu_v, omega )[i, j] \
                       + 2.0 * eta * d_c( v, w, omega )[j, i] * Nabla_f( nu_v, omega )[j, i])
      ) * sqrt_detg( omega ) * dx \
      - rho / 2.0 * ( \
                    ((w ** 2) * (n_lr( omega ))[i] * nu_v[i]) * sqrt_deth_lr( omega ) * (ds_l + ds_r) \
                    + ((w ** 2) * (n_tb( omega ))[i] * nu_v[i]) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
                    + ((w ** 2) * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
      ) \
      - ( \
                    (sigma * (n_lr( omega ))[i] * nu_v[i]) * sqrt_deth_lr( omega ) * (ds_l + ds_r) \
                    + (sigma * (n_tb( omega ))[i] * nu_v[i]) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
                    + (sigma * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
      ) \
      - 2.0 * eta * ( \
              (d_c( v, w, omega )[i, j] * g( omega )[i, k] * (n_lr( omega ))[k] * nu_v[j]) * sqrt_deth_lr( omega ) * ds_l \
              # BC (1g)^\omega is implemented here as a natural bc
              + (d_c( v, w, omega )[i, 1] * g( omega )[i, k] * (n_lr( omega ))[k] * nu_v[1]) * sqrt_deth_lr( omega ) * ds_r \
              + (d_c( v, w, omega )[i, j] * g( omega )[i, k] * (n_tb( omega ))[k] * nu_v[j]) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
              + (d_c( v, w, omega )[i, j] * g( omega )[i, k] * (n_circle( omega ))[k] * nu_v[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
      )

F_w = ( \
                    rho * (v[i] * v[k] * b( omega )[k, i]) * nu_w \
                    - rho * w * Nabla_v( vector_times_scalar( v, nu_w ), omega )[i, i] \
                    + 2.0 * kappa * ( \
                                  - g_c( omega )[i, j] * ((H( omega )).dx( i )) * (nu_w.dx( j )) \
                                  + 2.0 * H( omega ) * ((H( omega )) ** 2 - K( omega )) * nu_w \
                          ) \
                    - ( \
                                  2.0 * sigma * H( omega ) \
                                  + 2.0 * eta * (g_c( omega )[i, k] * Nabla_v( v, omega )[j, k] *
                                                 (b( omega ))[i, j] - 2.0 * w * (2.0 * (H( omega )) ** 2 - K( omega )))
                    ) * nu_w
      ) * sqrt_detg( omega ) * dx \
+ rho * ( \
              (w * nu_w * (n_lr( omega ))[j] * g( omega )[j, i] * v[i]) * sqrt_deth_lr( omega ) * (ds_l + ds_r) \
              + (w * nu_w * (n_tb( omega ))[j] * g( omega )[j, i] * v[i]) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
              + (w * nu_w * (n_circle( omega ))[j] * g( omega )[j, i] * v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
) \
+ 2.0 * kappa * ( \
              ( (n_lr( omega ))[i] * ((H( omega )).dx( i )) * nu_w ) * sqrt_deth_lr( omega ) * (ds_l + ds_r) \
              + ( (n_tb( omega ))[i] * ((H( omega )).dx( i )) * nu_w ) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
              + ( (n_circle( omega ))[i] * ((H( omega )).dx( i )) * nu_w ) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle
)

F_z = ( \
                    - w * ((normal( omega ))[2] - ((normal( omega ))[0] * omega[0] + (normal( omega ))[1] * omega[1])) * nu_z \
            ) * sqrt_detg( omega ) * dx



F_omega = ( z * Nabla_v( nu_omega, omega )[i, i] + omega[i] * nu_omega[i] ) * sqrt_detg( omega ) * dx \
          - ( \
                        ( (n_lr( omega ))[i] * g( omega )[i, j] * z * nu_omega[j] ) * sqrt_deth_lr( omega ) * (ds_l + ds_r) \
                        + ( (n_tb( omega ))[i] * g( omega )[i, j] * z * nu_omega[j] ) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
                        + ( (n_circle( omega ))[i] * g( omega )[i, j] * z * nu_omega[j] ) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_circle \
          )

F_N = alpha / r_mesh * ( \
 \
              + ( ( (n_tb(omega))[i] * g(omega)[i, j] * v[j] ) * ( (n_tb(omega))[k] * nu_v[k]) ) * sqrt_deth_tb( omega ) * (ds_t + ds_b) \
              + ( ( (n_circle(omega))[i] * g(omega)[i, j] * v[j] ) * ( (n_circle(omega))[k] * nu_v[k]) ) * sqrt_deth_circle(omega, c_r) * (1.0 / r) * ds_circle \
\
              + ( ( (n_lr(omega))[i] * omega[i] - omega_square ) * ((n_lr(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_lr( omega ) * ( ds_l + ds_r) \
              + ( ( (n_tb(omega))[i] * omega[i] - omega_square ) * ((n_tb(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_tb( omega ) * ( ds_t + ds_b) \
              + ( ( (n_circle(omega))[i] * omega[i] - omega_circle ) * ((n_circle(omega))[k] * g( omega )[k, l] * nu_omega[l]) ) * sqrt_deth_circle(omega, c_r) * (1.0 / r) * ds_circle \
 \
      )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega ) + F_N