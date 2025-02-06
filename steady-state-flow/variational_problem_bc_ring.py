from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_ring as rmsh


# CHANGE PARAMETERS HERE
v_r = 0.9950371902099891356653
v_R = 0.5
w_r = 0.0
w_R = 0.0
sigma_r = -0.9950248754694831
z_r = 1.0
z_R = 1.09900076985083984499716224302
omega_r = Constant(-0.099503719020998913567)
omega_R = Constant(0.095353867584048529241675292343)


#bending rigidity
kappa = 1.0
#density
rho = 1.0
#viscosity
eta = 1.0
#Nitche's parameter
alpha = 1e1

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
# CHANGE PARAMETERS HERE


fsp.v_0.interpolate(TangentVelocityExpression(element=fsp.Q_v.ufl_element()))
fsp.w_0.interpolate(NormalVelocityExpression(element=fsp.Q_w.ufl_element()))
fsp.sigma_0.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ))
fsp.omega_0.interpolate( OmegaExpression( element=fsp.Q_omega.ufl_element() ))
fsp.z_0.interpolate( ManifoldExpression( element=fsp.Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [v_0, w_0, sigma_0, omega_0, z_0])


# CHANGE PARAMETERS HERE
profile_v_r = Expression( ('v_r * x[0] / sqrt( pow(x[0], 2) + pow(x[1], 2) )', 'v_r * x[1] / sqrt( pow(x[0], 2) + pow(x[1], 2) )'),  v_r = v_r, element=Q.sub( 0 ).ufl_element() )
# CHANGE PARAMETERS HERE# CHANGE PARAMETERS HERE

# boundary conditions (BCs)
bc_v_r = DirichletBC( Q.sub( 0 ), profile_v_r, boundary_r )

# BCs for w_bar
bc_w_r = DirichletBC( Q.sub( 1 ), Constant( w_r ), boundary_r )
bc_w_R = DirichletBC( Q.sub( 1 ), Constant( w_R ), boundary_R )

#BC for sigma
bc_sigma_r = DirichletBC( Q.sub( 2 ), Constant( sigma_r ), boundary_r )

# BCs for z
bc_z_r = DirichletBC( Q.sub( 4 ), Expression( 'z_r', element=Q.sub( 4 ).ufl_element(), z_r=z_r), boundary_r )
bc_z_R = DirichletBC( Q.sub( 4 ), Expression( 'z_R', element=Q.sub( 4 ).ufl_element(), z_R=z_R), boundary_R )

# all BCs
bcs = [bc_v_r, bc_w_r, bc_w_R, bc_sigma_r, bc_z_r, bc_z_R]


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
                    + ((w ** 2) * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
                    + ((w ** 2) * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R
      ) \
      - ( \
                    + (sigma * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
                    + (sigma * (n_circle( omega ))[i] * nu_v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R
      ) \
      - 2.0 * eta * ( \
              + (d_c( v, w, omega )[i, j] * g( omega )[i, k] * (n_circle( omega ))[k] * nu_v[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
              + (d_c( v, w, omega )[i, j] * g( omega )[i, k] * (n_circle( omega ))[k] * nu_v[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R
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
              + (w * nu_w * (n_circle( omega ))[j] * g( omega )[j, i] * v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
              + (w * nu_w * (n_circle( omega ))[j] * g( omega )[j, i] * v[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R
) \
+ 2.0 * kappa * ( \
              + ( (n_circle( omega ))[i] * ((H( omega )).dx( i )) * nu_w ) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
              + ( (n_circle( omega ))[i] * ((H( omega )).dx( i )) * nu_w ) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R \
)

F_z = ( \
                    - w * ((normal( omega ))[2] - ((normal( omega ))[0] * omega[0] + (normal( omega ))[1] * omega[1])) * nu_z \
            ) * sqrt_detg( omega ) * dx

F_omega = (z * Nabla_v( nu_omega, omega )[i, i] + omega[i] * nu_omega[i]) * sqrt_detg( omega ) * dx \
          - ( \
                      + ((n_circle( omega ))[i] * g( omega )[i, j] * z * nu_omega[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
                      + ((n_circle( omega ))[i] * g( omega )[i, j] * z * nu_omega[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R \
              )

F_N = alpha / r_mesh * ( \
            + (((n_circle( omega ))[i] * omega[i] - omega_r) * ((n_circle( omega ))[k] * g( omega )[k, l] * nu_omega[l])) * sqrt_deth_circle( omega, c_r ) * ds_r \
            + (((n_circle( omega ))[i] * omega[i] - omega_R) * ((n_circle( omega ))[k] * g( omega )[k, l] * nu_omega[l])) * sqrt_deth_circle( omega, c_r ) * ds_R \
 \
            + ((n_circle( omega )[i] * g( omega )[i, j] * v[j] - v_R) * (n_circle( omega )[k] * nu_v[k])) * sqrt_deth_circle( omega, c_r ) * ds_R \
    )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega ) + F_N