from fenics import *
import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_ring as rmsh

i, j, k, l = ufl.indices( 4 )



# CHANGE PARAMETERS HERE


#bending rigidity
kappa = 1.0
#density
rho = 1.0
#viscosity
eta = 1.0
#Nitche's parameter
alpha = 1e1
CC = 0.1

v_r_const = 0.9950371902099891356653
w_R_const = 0.0
sigma_r_const = -0.99502487546948322183
z_R_const = 1.09900076963490661833037160663
omega_r_const = 0.100000000000000000000
omega_R_const = 0.095790340774569785068426326150

class v_r_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = v_r_const * x[0] / geo.my_norm( x )
        values[1] = v_r_const * x[1] / geo.my_norm( x )

    def value_shape(self):
        return (2,)

class w_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = w_R_const

    def value_shape(self):
        return (1,)

class sigma_r_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = sigma_r_const

    def value_shape(self):
        return (1,)


class z_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = z_R_const

    def value_shape(self):
        return (1,)


class omega_r_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = omega_r_const * x[0] / geo.my_norm(x)
        values[1] = omega_r_const * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)

class omega_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = omega_R_const * x[0] / geo.my_norm(x)
        values[1] = omega_R_const * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)

class mu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0


    def value_shape(self):
        return (1,)


# CHANGE PARAMETERS HERE

v_r = interpolate( v_r_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )
w_R = interpolate( w_R_Expression( element=fsp.Q_w.ufl_element() ), fsp.Q_w )
sigma_r = interpolate( sigma_r_Expression( element=fsp.Q_sigma.ufl_element() ), fsp.Q_sigma )
z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_r = interpolate( omega_r_Expression( element=fsp.Q_omega.ufl_element() ), fsp.Q_omega )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_omega.ufl_element() ), fsp.Q_omega )

mu_exact = interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ), fsp.Q_mu )


fsp.v_0.interpolate( v_r_Expression( element=fsp.Q_v.ufl_element() ) )
fsp.w_0.interpolate( w_R_Expression( element=fsp.Q_w.ufl_element() ) )
fsp.sigma_0.interpolate( sigma_r_Expression( element=fsp.Q_sigma.ufl_element() ) )
fsp.z_0.interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_0.interpolate( omega_r_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_0.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ))

# fsp.nu_0.interpolate( NuExpression( element=fsp.Q_nu.ufl_element() ) )
# fsp.tau_0.interpolate( TauExpression( element=fsp.Q_tau.ufl_element() ) )


#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# fsp.assigner.assign(fsp.psi, [fsp.v_0, fsp.w_0, fsp.sigma_0,  fsp.z_0, fsp.omega_0, fsp.mu_0])


# CHANGE PARAMETERS HERE
# profile_v_r = Expression( ('v_r * x[0] / sqrt( pow(x[0], 2) + pow(x[1], 2) )', 'v_r * x[1] / sqrt( pow(x[0], 2) + pow(x[1], 2) )'), v_r = v_r_const, element=fsp.Q.sub( 0 ).ufl_element() )
# CHANGE PARAMETERS HERE# CHANGE PARAMETERS HERE

# boundary conditions (BCs)
bc_v_r = DirichletBC( fsp.Q.sub( 0 ), v_r, rmsh.boundary_r )
bc_w_R = DirichletBC( fsp.Q.sub( 1 ), w_R, rmsh.boundary_R )
bc_sigma_r = DirichletBC( fsp.Q.sub( 2 ), sigma_r, rmsh.boundary_r )
bc_z_R = DirichletBC( fsp.Q.sub( 3 ), z_R, rmsh.boundary_R )
bc_omega_r = DirichletBC( fsp.Q.sub( 4 ), omega_r, rmsh.boundary_r )
bc_omega_R = DirichletBC( fsp.Q.sub( 4 ), omega_R, rmsh.boundary_R )

# all BCs
bcs = [bc_v_r, bc_w_R, bc_sigma_r, bc_z_R, bc_omega_r, bc_omega_R]

# Define variational problem : F_v, F_z are related to the PDEs for v, ..., z respectively . F_N enforces the BCs with Nitsche's method.
# To be safe, I explicitly wrote each term on each part of the boundary with its own normal vector and pull-back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle,
# the surface elements are ds_l + ds_r, and the normal is n_lr(omega), and the pull-back of the metric is sqrt_deth_lr: this avoids odd interpolations at the corners of the rectangle edges.

F_sigma = (geo.Nabla_v( fsp.v, fsp.omega )[i, i] - 2.0 * fsp.mu * fsp.w) * fsp.nu_sigma * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_v = ( \
                    rho * ( \
                          (fsp.v[j] * geo.Nabla_v( fsp.v,  fsp.omega )[i, j] - 2.0 * fsp.v[j] * fsp.w * geo.g_c( fsp.omega )[i, k] * geo.b( fsp.omega )[k, j]) * fsp.nu_v[i] \
                          + 1.0 / 2.0 * (fsp.w ** 2) * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                  ) \
                    + (fsp.sigma * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                       + 2.0 * eta * geo.d_c( fsp.v,  fsp.w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu_v, fsp.omega )[j, i])
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - rho / 2.0 * ( \
                    + ((fsp.w ** 2) * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
                    + ((fsp.w ** 2) * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
      ) \
      - ( \
                    + (fsp.sigma * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
                    + (fsp.sigma * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
      ) \
      - 2.0 * eta * ( \
              + (geo.d_c( fsp.v,  fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
              + (geo.d_c( fsp.v,  fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
      )

F_w = ( \
                    rho * (fsp.v[i] * fsp.v[k] * geo.b( fsp.omega )[k, i]) * fsp.nu_w \
                    - rho * fsp.w * geo.Nabla_v( geo.vector_times_scalar( fsp.v,  fsp.nu_w ), fsp.omega )[i, i] \
                    + 2.0 * kappa * ( \
                                  - geo.g_c( fsp.omega )[i, j] * (fsp.mu.dx( i )) * (fsp.nu_w.dx( j )) \
                                  + 2.0 * fsp.mu * (fsp.mu ** 2 - geo.K( fsp.omega )) * fsp.nu_w \
                          ) \
                    - ( \
                                  2.0 * fsp.sigma * fsp.mu \
                                  + 2.0 * eta * (geo.g_c( fsp.omega )[i, k] * geo.Nabla_v( fsp.v,  fsp.omega )[j, k] *
                                                 (geo.b( fsp.omega ))[i, j] - 2.0 * fsp.w * (2.0 * fsp.mu ** 2 - geo.K( fsp.omega )))
                    ) * fsp.nu_w
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
+ rho * ( \
              + (fsp.w * fsp.nu_w * (bgeo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
              + (fsp.w * fsp.nu_w * (bgeo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
) \
+ 2.0 * kappa * ( \
              + ( (bgeo.n_circle( fsp.omega ))[i] * (fsp.mu.dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
              + ( (bgeo.n_circle( fsp.omega ))[i] * (fsp.mu.dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R \
)

F_z = ( \
                    - fsp.w * ((geo.normal( fsp.omega ))[2] - ((geo.normal( fsp.omega ))[0] * fsp.omega[0] + (geo.normal( fsp.omega ))[1] * fsp.omega[1])) * fsp.nu_z \
            ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_omega = (fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] + fsp.omega[i] * fsp.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          - ( \
                      + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
                      + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R \
              )

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx


F_N = alpha / rmsh.r_mesh * ( \
            # + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_exact) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * rmsh.ds_r \
            # + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - (bgeo.n_circle(omega_exact ))[i] * omega_exact[i]) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * rmsh.ds_R \ \
 \
            + ((bgeo.n_circle( fsp.omega )[i] * geo.g( fsp.omega )[i, j] * fsp.v[j] - bgeo.n_circle( fsp.omega )[i] * geo.g( fsp.omega )[i, j] * v_r[j]) * (bgeo.n_circle( fsp.omega )[k] * fsp.nu_v[k])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * rmsh.ds_R \
    )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega + F_mu) + F_N


#post-processing variational functional
F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R


F_pp_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R

F_pp_d = ((geo.d(fsp.v, fsp.w, fsp.omega)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx
