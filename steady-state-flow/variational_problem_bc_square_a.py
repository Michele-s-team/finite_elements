from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )


# CHANGE PARAMETERS HERE
v_l = 1.0
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
omega_circle = interpolate( omega_circle_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_square = interpolate( omega_square_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.v_0.interpolate(TangentVelocityExpression(element=fsp.Q_v.ufl_element()))
fsp.w_0.interpolate(NormalVelocityExpression(element=fsp.Q_w.ufl_element()))
fsp.sigma_0.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ))
fsp.omega_0.interpolate( OmegaExpression( element=fsp.Q_omega.ufl_element() ))
fsp.z_0.interpolate( ManifoldExpression( element=fsp.Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [v_0, w_0, sigma_0, omega_0, z_0])


# CHANGE PARAMETERS HERE
l_profile_v = Expression( ('v_l', '0'), v_l=v_l, element = fsp.Q_v.ufl_element() )
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

# Define variational problem : F_v, F_z are related to the PDEs for v, ..., z respectively . F_N enforces the BCs with Nitsche's method.
# To be safe, I explicitly wrote each term on each part of the boundary with its own normal vector and pull-back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle,
# the surface elements are ds_l + ds_r, and the normal is n_lr(omega), and the pull-back of the metric is sqrt_deth_lr: this avoids odd interpolations at the corners of the rectangle edges.

F_sigma = (geo.Nabla_v( v, fsp.omega )[i, i] - 2.0 * H( fsp.omega ) * w) * fsp.nu_sigma * bgeo.sqrt_detg( fsp.omega ) * rmsh.dx

F_v = ( \
                    rho * ( \
                          (v[j] * geo.Nabla_v( v, fsp.omega )[i, j] - 2.0 * v[j] * w * geo.g_c( fsp.omega )[i, k] * geo.b( fsp.omega )[k, j]) * fsp.nu_v[i] \
                          + 1.0 / 2.0 * (w ** 2) * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                  ) \
                    + (sigma * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                       + 2.0 * eta * d_c( v, w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu_v, fsp.omega )[j, i])
      ) * bgeo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - rho / 2.0 * ( \
                    ((w ** 2) * (n_lr( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                    + ((w ** 2) * (n_tb( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                    + ((w ** 2) * (n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle
      ) \
      - ( \
                    (sigma * (n_lr( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                    + (sigma * (n_tb( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                    + (sigma * (n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle
      ) \
      - 2.0 * eta * ( \
              (d_c( v, w, fsp.omega )[i, j] * g( fsp.omega )[i, k] * (n_lr( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_l \
              # BC (1g)^\omega is implemented here as a natural bc
              + (d_c( v, w, fsp.omega )[i, 1] * g( fsp.omega )[i, k] * (n_lr( fsp.omega ))[k] * fsp.nu_v[1]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_r \
              + (d_c( v, w, fsp.omega )[i, j] * g( fsp.omega )[i, k] * (n_tb( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + (d_c( v, w, fsp.omega )[i, j] * g( fsp.omega )[i, k] * (n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle
      )

F_w = ( \
                    rho * (v[i] * v[k] * geo.b( fsp.omega )[k, i]) * fsp.nu_w \
                    - rho * w * geo.Nabla_v( vector_times_scalar( v, fsp.nu_w ), fsp.omega )[i, i] \
                    + 2.0 * kappa * ( \
                                  - geo.g_c( fsp.omega )[i, j] * ((H( fsp.omega )).dx( i )) * (fsp.nu_w.dx( j )) \
                                  + 2.0 * H( fsp.omega ) * ((H( fsp.omega )) ** 2 - K( fsp.omega )) * fsp.nu_w \
                          ) \
                    - ( \
                                  2.0 * sigma * H( fsp.omega ) \
                                  + 2.0 * eta * (geo.g_c( fsp.omega )[i, k] * geo.Nabla_v( v, fsp.omega )[j, k] *
                                                 (geo.b( fsp.omega ))[i, j] - 2.0 * w * (2.0 * (H( fsp.omega )) ** 2 - K( fsp.omega )))
                    ) * fsp.nu_w
      ) * bgeo.sqrt_detg( fsp.omega ) * rmsh.dx \
+ rho * ( \
              (w * fsp.nu_w * (n_lr( fsp.omega ))[j] * g( fsp.omega )[j, i] * v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
              + (w * fsp.nu_w * (n_tb( fsp.omega ))[j] * g( fsp.omega )[j, i] * v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + (w * fsp.nu_w * (n_circle( fsp.omega ))[j] * g( fsp.omega )[j, i] * v[i]) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle
) \
+ 2.0 * kappa * ( \
              ( (n_lr( fsp.omega ))[i] * ((H( fsp.omega )).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
              + ( (n_tb( fsp.omega ))[i] * ((H( fsp.omega )).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + ( (n_circle( fsp.omega ))[i] * ((H( fsp.omega )).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle
)

F_z = ( \
                    - w * ((normal( fsp.omega ))[2] - ((normal( fsp.omega ))[0] * fsp.omega[0] + (normal( fsp.omega ))[1] * fsp.omega[1])) * fsp.nu_z \
            ) * bgeo.sqrt_detg( fsp.omega ) * rmsh.dx



F_omega = ( z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] + fsp.omega[i] * fsp.nu_omega[i] ) * bgeo.sqrt_detg( fsp.omega ) * rmsh.dx \
          - ( \
                        ( (n_lr( fsp.omega ))[i] * g( fsp.omega )[i, j] * z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                        + ( (n_tb( fsp.omega ))[i] * g( fsp.omega )[i, j] * z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                        + ( (n_circle( fsp.omega ))[i] * g( fsp.omega )[i, j] * z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_circle( fsp.omega, c_r ) * (1.0 / r) * rmsh.ds_circle \
          )

F_N = alpha / r_mesh * ( \
 \
              + ( ( (n_tb(fsp.omega))[i] * g(fsp.omega)[i, j] * v[j] ) * ( (n_tb(fsp.omega))[k] * fsp.nu_v[k]) ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + ( ( (n_circle(fsp.omega))[i] * g(fsp.omega)[i, j] * v[j] ) * ( (n_circle(fsp.omega))[k] * fsp.nu_v[k]) ) * bgeo.sqrt_deth_circle(fsp.omega, c_r) * (1.0 / r) * rmsh.ds_circle \
\
              + ( ( (n_lr(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((n_lr(fsp.omega))[k] * g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_lr( fsp.omega ) * ( rmsh.ds_l + rmsh.ds_r) \
              + ( ( (n_tb(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((n_tb(fsp.omega))[k] * g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_tb( fsp.omega ) * ( rmsh.ds_t + rmsh.ds_b) \
              + ( ( (n_circle(fsp.omega))[i] * fsp.omega[i] - omega_circle ) * ((n_circle(fsp.omega))[k] * g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_circle(fsp.omega, c_r) * (1.0 / r) * rmsh.ds_circle \
 \
      )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega ) + F_N