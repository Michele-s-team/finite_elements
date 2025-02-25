from fenics import *
import numpy as np
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )


# CHANGE PARAMETERS HERE
v_l_const = 1.0
w_square_const = 0.0
sigma_r_const = 0.0
z_square_const = 0.0
omega_circle_const = -0.1
omega_square_const = 0.0
#bending rigidity
kappa = 1.0
#density
rho = 1.0
#viscosity
eta = 1.0
#Nitche's parameter
alpha = 1e1

class v_l_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = v_l_const
        values[1] = 0

    def value_shape(self):
        return (2,)

class v_circle_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

class w_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = w_square_const

    def value_shape(self):
        return (1,)

class sigma_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = sigma_r_const

    def value_shape(self):
        return (1,)


class z_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = z_square_const

    def value_shape(self):
        return (1,)

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
        values[0] = omega_circle_const * (x[0] - rmsh.c_r[0])/geo.my_norm(x-rmsh.c_r)
        values[1] = omega_circle_const * (x[1] - rmsh.c_r[0])/geo.my_norm(x-rmsh.c_r)

    def value_shape(self):
        return (2,)


class omega_square_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_square_const

    def value_shape(self):
        return (1,)


# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_circle = interpolate( omega_circle_Expression( element=fsp.Q_omega.ufl_element() ), fsp.Q_omega )
omega_square = interpolate( omega_square_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.v_0.interpolate(TangentVelocityExpression(element=fsp.Q_v.ufl_element()))
fsp.w_0.interpolate(NormalVelocityExpression(element=fsp.Q_w.ufl_element()))
fsp.sigma_0.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ))
fsp.omega_0.interpolate( OmegaExpression( element=fsp.Q_omega.ufl_element() ))
fsp.z_0.interpolate( ManifoldExpression( element=fsp.Q_z.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [v_0, w_0, sigma_0, omega_0, z_0])

v_l = interpolate( v_l_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )
v_circle = interpolate( v_circle_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )
w_square = interpolate( w_square_Expression( element=fsp.Q_w.ufl_element() ), fsp.Q_w )
sigma_r = interpolate( sigma_r_Expression( element=fsp.Q_sigma.ufl_element() ), fsp.Q_sigma )

z_square = interpolate( z_square_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )


# boundary conditions (BCs)
# BCs for v_bar
bc_v_l = DirichletBC( fsp.Q.sub( 0 ), v_l, rmsh.boundary_l )
bc_v_circle = DirichletBC( fsp.Q.sub( 0 ), v_circle, rmsh.boundary_circle )

# BCs for w_bar
bc_w_square = DirichletBC( fsp.Q.sub( 1 ), w_square, rmsh.boundary_square )

#BC for sigma
bc_sigma_r = DirichletBC( fsp.Q.sub( 2 ), sigma_r, rmsh.boundary_r )

# CHANGE PARAMETERS HERE
# BCs for z and omega
bc_z_square = DirichletBC( fsp.Q.sub( 3 ), z_square, rmsh.boundary_square )
bc_omega_circle = DirichletBC( fsp.Q.sub( 4 ), omega_circle, rmsh.boundary_circle )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_v_l, bc_v_circle, bc_w_square, bc_sigma_r, bc_z_square, bc_omega_circle]

# Define variational problem : F_v, F_z are related to the PDEs for v, ..., z respectively . F_N enforces the BCs with Nitsche's method.
# To be safe, I explicitly wrote each term on each part of the boundary with its own normal vector and pull-back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle,
# the surface elements are ds_l + ds_r, and the normal is n_lr(omega), and the pull-back of the metric is sqrt_deth_lr: this avoids odd interpolations at the corners of the rectangle edges.

F_sigma = (geo.Nabla_v( fsp.v, fsp.omega )[i, i] - 2.0 * fsp.mu * fsp.w) * fsp.nu_sigma * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_v = ( \
                    rho * ( \
                          (fsp.v[j] * geo.Nabla_v( fsp.v, fsp.omega )[i, j] - 2.0 * fsp.v[j] * fsp.w * geo.g_c( fsp.omega )[i, k] * geo.b( fsp.omega )[k, j]) * fsp.nu_v[i] \
                          + 1.0 / 2.0 * (fsp.w ** 2) * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                  ) \
                    + (fsp.sigma * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                       + 2.0 * eta * geo.d_c( fsp.v, fsp.w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu_v, fsp.omega )[j, i])
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - rho / 2.0 * ( \
                    ((fsp.w ** 2) * (bgeo.n_lr( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                    + ((fsp.w ** 2) * (bgeo.n_tb( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                    + ((fsp.w ** 2) * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
      ) \
      - ( \
                    (fsp.sigma * (bgeo.n_lr( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                    + (fsp.sigma * (bgeo.n_tb( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                    + (fsp.sigma * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
      ) \
      - 2.0 * eta * ( \
              (geo.d_c( fsp.v, fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_lr( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_l \
              # BC (1g)^\omega is implemented here as a natural bc
              + (geo.d_c( fsp.v, fsp.w, fsp.omega )[i, 1] * geo.g( fsp.omega )[i, k] * (bgeo.n_lr( fsp.omega ))[k] * fsp.nu_v[1]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_r \
              + (geo.d_c( fsp.v, fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_tb( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + (geo.d_c( fsp.v, fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
      )

F_w = ( \
                    rho * (fsp.v[i] * fsp.v[k] * geo.b( fsp.omega )[k, i]) * fsp.nu_w \
                    - rho * fsp.w * geo.Nabla_v( geo.vector_times_scalar( fsp.v, fsp.nu_w ), fsp.omega )[i, i] \
                    + 2.0 * kappa * ( \
                                  - geo.g_c( fsp.omega )[i, j] * ((fsp.mu).dx( i )) * (fsp.nu_w.dx( j )) \
                                  + 2.0 * fsp.mu * ((fsp.mu) ** 2 - geo.K( fsp.omega )) * fsp.nu_w \
                          ) \
                    - ( \
                                  2.0 * fsp.sigma * fsp.mu \
                                  + 2.0 * eta * (geo.g_c( fsp.omega )[i, k] * geo.Nabla_v( fsp.v, fsp.omega )[j, k] *
                                                 (geo.b( fsp.omega ))[i, j] - 2.0 * fsp.w * (2.0 * (fsp.mu) ** 2 - geo.K( fsp.omega )))
                    ) * fsp.nu_w
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
+ rho * ( \
              (fsp.w * fsp.nu_w * (bgeo.n_lr( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
              + (fsp.w * fsp.nu_w * (bgeo.n_tb( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + (fsp.w * fsp.nu_w * (bgeo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
) \
+ 2.0 * kappa * ( \
              ( (bgeo.n_lr( fsp.omega ))[i] * ((fsp.mu).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
              + ( (bgeo.n_tb( fsp.omega ))[i] * ((fsp.mu).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              + ( (bgeo.n_circle( fsp.omega ))[i] * ((fsp.mu).dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle
)

F_z = ( \
                    - fsp.w * ((geo.normal( fsp.omega ))[2] - ((geo.normal( fsp.omega ))[0] * fsp.omega[0] + (geo.normal( fsp.omega ))[1] * fsp.omega[1])) * fsp.nu_z \
            ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx



F_omega = ( fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] + fsp.omega[i] * fsp.nu_omega[i] ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          - ( \
                        ( (bgeo.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_lr( fsp.omega ) * (rmsh.ds_l + rmsh.ds_r) \
                        + ( (bgeo.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
                        + ( (bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle \
          )

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx


F_N = alpha / rmsh.r_mesh * ( \
 \
              + ( ( (bgeo.n_tb(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.v[j] ) * ( (bgeo.n_tb(fsp.omega))[k] * fsp.nu_v[k]) ) * bgeo.sqrt_deth_tb( fsp.omega ) * (rmsh.ds_t + rmsh.ds_b) \
              # + ( ( (bgeo.n_circle(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.v[j] ) * ( (bgeo.n_circle(fsp.omega))[k] * fsp.nu_v[k]) ) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle \
\
              + ( ( (bgeo.n_lr(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((bgeo.n_lr(fsp.omega))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_lr( fsp.omega ) * ( rmsh.ds_l + rmsh.ds_r) \
              + ( ( (bgeo.n_tb(fsp.omega))[i] * fsp.omega[i] - omega_square ) * ((bgeo.n_tb(fsp.omega))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_tb( fsp.omega ) * ( rmsh.ds_t + rmsh.ds_b) \
              # + ( ( (bgeo.n_circle(fsp.omega))[i] * fsp.omega[i] - omega_circle ) * ((bgeo.n_circle(fsp.omega))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l]) ) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle \
 \
      )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega + F_mu ) + F_N


#post-processing variational functional
F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
       - ((bgeo.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \


F_pp_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
           - ((bgeo.n_lr( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
           - ((bgeo.n_tb( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r


F_pp_d = ((geo.d(fsp.v, fsp.w, fsp.omega)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx
