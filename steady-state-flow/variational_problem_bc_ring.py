from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_ring as rmsh

i, j, k, l = ufl.indices( 4 )



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

F_sigma = (geo.Nabla_v( v, fsp.omega )[i, i] - 2.0 * geo.H( fsp.omega ) * w) * geo.nu_sigma * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_v = ( \
                    rho * ( \
                          (v[j] * geo.Nabla_v( v, fsp.omega )[i, j] - 2.0 * v[j] * w * geo.g_c( fsp.omega )[i, k] * geo.b( fsp.omega )[k, j]) * geo.nu_v[i] \
                          + 1.0 / 2.0 * (w ** 2) * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( geo.nu_v, fsp.omega )[i, j] \
                  ) \
                    + (fsp.sigma * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( geo.nu_v, fsp.omega )[i, j] \
                       + 2.0 * eta * geo.d_c( v, w, fsp.omega )[j, i] * geo.Nabla_f( geo.nu_v, fsp.omega )[j, i])
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - rho / 2.0 * ( \
                    + ((w ** 2) * (geo.n_circle( fsp.omega ))[i] * geo.nu_v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
                    + ((w ** 2) * (geo.n_circle( fsp.omega ))[i] * geo.nu_v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R
      ) \
      - ( \
                    + (fsp.sigma * (geo.n_circle( fsp.omega ))[i] * geo.nu_v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
                    + (fsp.sigma * (geo.n_circle( fsp.omega ))[i] * geo.nu_v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R
      ) \
      - 2.0 * eta * ( \
              + (geo.d_c( v, w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (geo.n_circle( fsp.omega ))[k] * geo.nu_v[j]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
              + (geo.d_c( v, w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (geo.n_circle( fsp.omega ))[k] * geo.nu_v[j]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R
      )

F_w = ( \
                    rho * (v[i] * v[k] * geo.b( fsp.omega )[k, i]) * geo.nu_w \
                    - rho * w * geo.Nabla_v( vector_times_scalar( v, geo.nu_w ), fsp.omega )[i, i] \
                    + 2.0 * kappa * ( \
                                  - geo.g_c( fsp.omega )[i, j] * ((geo.H( fsp.omega )).dx( i )) * (geo.nu_w.dx( j )) \
                                  + 2.0 * geo.H( fsp.omega ) * ((geo.H( fsp.omega )) ** 2 - K( fsp.omega )) * geo.nu_w \
                          ) \
                    - ( \
                                  2.0 * fsp.sigma * geo.H( fsp.omega ) \
                                  + 2.0 * eta * (geo.g_c( fsp.omega )[i, k] * geo.Nabla_v( v, fsp.omega )[j, k] *
                                                 (geo.b( fsp.omega ))[i, j] - 2.0 * w * (2.0 * (geo.H( fsp.omega )) ** 2 - K( fsp.omega )))
                    ) * geo.nu_w
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
+ rho * ( \
              + (w * geo.nu_w * (geo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
              + (w * geo.nu_w * (geo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * v[i]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R
) \
+ 2.0 * kappa * ( \
              + ( (geo.n_circle( fsp.omega ))[i] * ((geo.H( fsp.omega )).dx( i )) * geo.nu_w ) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
              + ( (geo.n_circle( fsp.omega ))[i] * ((geo.H( fsp.omega )).dx( i )) * geo.nu_w ) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R \
)

F_z = ( \
                    - w * ((geo.normal( fsp.omega ))[2] - ((geo.normal( fsp.omega ))[0] * fsp.omega[0] + (geo.normal( fsp.omega ))[1] * fsp.omega[1])) * geo.nu_z \
            ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_omega = (z * geo.Nabla_v( geo.nu_omega, fsp.omega )[i, i] + fsp.omega[i] * geo.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          - ( \
                      + ((geo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * z * geo.nu_omega[j]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * geo.ds_r \
                      + ((geo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * z * geo.nu_omega[j]) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * geo.ds_R \
              )

F_N = alpha / rmsh.r_mesh * ( \
            + (((geo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_r) * ((geo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * geo.nu_omega[l])) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * geo.ds_r \
            + (((geo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_R) * ((geo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * geo.nu_omega[l])) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * geo.ds_R \
 \
            + ((geo.n_circle( fsp.omega )[i] * geo.g( fsp.omega )[i, j] * v[j] - v_R) * (geo.n_circle( fsp.omega )[k] * geo.nu_v[k])) * geo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * geo.ds_R \
    )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega ) + F_N